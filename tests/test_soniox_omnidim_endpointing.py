#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fork-local Soniox endpointing behaviour.

Three things this fork does differently from upstream, kept in their own file
so upstream's ``test_soniox_stt.py`` can be taken wholesale on the next
upgrade:

1. Server-side endpoint detection stays enabled even with
   ``vad_force_turn_endpoint=True``, so committed tokens still flush when the
   pipeline VAD never fires a stop (soft speech under bot audio, where VAD
   thresholds are raised).
2. Because of (1), the endpoint tuning settings reach the wire in that mode —
   upstream drops them there.
3. Interim transcriptions are deduped: Soniox re-sends the unchanged buffer
   roughly every second, and re-pushing it downstream reads as continuous user
   activity (re-triggered turn starts, a starved turn-stop watchdog, a
   perpetually reset idle timer).

The matching "and it emits no user-speaking frames of its own" invariant is
covered by upstream's ``test_pipecat_mode_*`` tests. It is load-bearing for
this app: consumers treat a bare ``UserStartedSpeakingFrame`` as already
word-gated, so one emitted by the STT would cancel an armed hangup and bypass
min-words barge-in gating.
"""

import json
from unittest.mock import AsyncMock

import pytest
from websockets.protocol import State

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.services.soniox.stt import END_TOKEN, SonioxSTTService


class _FakeWebsocket:
    def __init__(self, messages=(), *, state=State.OPEN):
        self._messages = list(messages)
        self.state = state
        self.send = AsyncMock()

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


async def _connect_and_capture_config(monkeypatch, service):
    """Run the connect handshake and return the config dict sent to Soniox."""
    fake_ws = _FakeWebsocket()

    async def fake_websocket_connect(*args, **kwargs):
        return fake_ws

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )
    monkeypatch.setattr(service, "_call_event_handler", AsyncMock())

    await service._connect_websocket()

    assert fake_ws.send.await_count == 1, "expected exactly one config message"
    return json.loads(fake_ws.send.await_args.args[0])


def _pushing_service(monkeypatch, pushed, **kwargs):
    """A service whose pushed frames are recorded as (type, text)."""
    service = SonioxSTTService(api_key="test-key", **kwargs)

    async def fake_push_frame(frame, direction=None):
        pushed.append((type(frame), getattr(frame, "text", None)))

    async def fake_noop(*args, **kwargs):
        pass

    monkeypatch.setattr(service, "push_frame", fake_push_frame)
    monkeypatch.setattr(service, "_handle_transcription", fake_noop)
    monkeypatch.setattr(service, "start_processing_metrics", fake_noop)
    monkeypatch.setattr(service, "stop_processing_metrics", fake_noop)
    monkeypatch.setattr(service, "emit_stt_usage_metrics", fake_noop)
    return service


@pytest.mark.asyncio
async def test_endpoint_detection_stays_on_in_pipecat_mode(monkeypatch):
    """Upstream sets ``enable_endpoint_detection = not vad_force_turn_endpoint``,
    which switches the backstop off in the mode this app runs."""
    service = SonioxSTTService(api_key="test-key")  # vad_force_turn_endpoint=True default
    config = await _connect_and_capture_config(monkeypatch, service)
    assert config["enable_endpoint_detection"] is True


@pytest.mark.asyncio
async def test_endpoint_settings_reach_the_wire_in_pipecat_mode(monkeypatch):
    """The production endpointing knobs must survive to the connect message;
    they are dead configuration if Soniox never receives them."""
    service = SonioxSTTService(
        api_key="test-key",
        settings=SonioxSTTService.Settings(
            max_endpoint_delay_ms=1500,
            endpoint_sensitivity=0.3,
            endpoint_latency_adjustment_level=2,
            max_non_final_tokens_duration_ms=1200,
        ),
    )
    config = await _connect_and_capture_config(monkeypatch, service)

    assert config["max_endpoint_delay_ms"] == 1500
    assert config["endpoint_sensitivity"] == 0.3
    assert config["endpoint_latency_adjustment_level"] == 2
    assert config["max_non_final_tokens_duration_ms"] == 1200


@pytest.mark.asyncio
async def test_unchanged_interim_is_not_pushed_again(monkeypatch):
    """Soniox re-sends the same non-final buffer on a timer; only a changed
    transcript is user activity."""
    pushed = []
    service = _pushing_service(monkeypatch, pushed)
    service._websocket = _FakeWebsocket(
        [
            json.dumps({"tokens": [{"text": "Hel", "is_final": False}]}),
            # Same text again — a keepalive-ish repeat, not new speech.
            json.dumps({"tokens": [{"text": "Hel", "is_final": False}]}),
            json.dumps({"tokens": [{"text": "Hello", "is_final": False}]}),
        ]
    )

    await service._receive_messages()

    assert pushed == [
        (InterimTranscriptionFrame, "Hel"),
        (InterimTranscriptionFrame, "Hello"),
    ]


@pytest.mark.asyncio
async def test_dedupe_resets_after_a_final_so_a_repeated_phrase_still_pushes(monkeypatch):
    """The buffer empties on the endpoint flush, so the next utterance must be
    able to push the same text the previous one ended on."""
    pushed = []
    service = _pushing_service(monkeypatch, pushed)
    service._websocket = _FakeWebsocket(
        [
            json.dumps({"tokens": [{"text": "yes", "is_final": False}]}),
            json.dumps(
                {"tokens": [{"text": "yes", "is_final": True}, {"text": END_TOKEN, "is_final": True}]}
            ),
            # A new utterance with identical wording.
            json.dumps({"tokens": [{"text": "yes", "is_final": False}]}),
        ]
    )

    await service._receive_messages()

    assert pushed == [
        (InterimTranscriptionFrame, "yes"),
        (TranscriptionFrame, "yes"),
        (InterimTranscriptionFrame, "yes"),
    ]
