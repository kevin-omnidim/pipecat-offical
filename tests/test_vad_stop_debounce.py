#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""``VADAnalyzer.stop_debounce_secs``: the silence the analyzer really counts
before declaring a stop.

``params.stop_secs`` is the *reported* value (turn-stop strategies subtract it
from the STT finalize window); ``_vad_stop_frames`` is what actually debounces,
and a runtime debounce change may move the counter without moving the report.
The STT TTFB clock needs the real speech end, so it needs the counted value.
"""

import pytest

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState
from pipecat.frames.frames import InputAudioRawFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.tests.utils import run_test


class _CountingVAD(VADAnalyzer):
    def __init__(self, **params):
        super().__init__(sample_rate=16000, params=VADParams(**params))

    def num_frames_required(self) -> int:
        return 512  # 32 ms blocks at 16 kHz

    def voice_confidence(self, buffer: bytes) -> float:
        return 0.0

    async def analyze_audio(self, buffer: bytes) -> VADState:
        return VADState.QUIET


def test_stop_debounce_equals_reported_stop_secs_until_the_counter_moves():
    vad = _CountingVAD(stop_secs=0.2)
    vad.set_sample_rate(16000)

    # 0.2 s at 32 ms blocks rounds to 6 blocks = 0.192 s: the counted value,
    # not the configured one.
    assert vad.stop_debounce_secs == pytest.approx(6 * 512 / 16000)


def test_stop_debounce_follows_the_counter_not_the_report():
    vad = _CountingVAD(stop_secs=0.2)
    vad.set_sample_rate(16000)
    block_secs = 512 / 16000

    # Runtime debounce change: +0.4 s of frames, report left at 0.2 s.
    vad._vad_stop_frames += round(0.4 / block_secs)

    assert vad.params.stop_secs == 0.2
    assert vad.stop_debounce_secs == pytest.approx((6 + round(0.4 / block_secs)) * block_secs)


def test_stop_debounce_is_zero_before_the_analyzer_is_started():
    vad = _CountingVAD(stop_secs=0.2)
    # No sample rate yet → no block size → nothing counted. An analyzer in
    # this state has also produced no stop, so nothing reads the value.
    assert vad.stop_debounce_secs == 0.0


#
# Producers: every VADUserStoppedSpeakingFrame carries the counted debounce,
# while the reported stop_secs stays exactly what params say.
#

EXTRA_STOP_BLOCKS = 12  # +0.384 s of counted debounce at 32 ms blocks


class _BumpedVAD(VADAnalyzer):
    """Speaks on the first block, goes quiet on the second, and — like a live
    analyzer whose debounce was raised at runtime — counts more stop frames
    than ``params.stop_secs`` reports."""

    def __init__(self):
        super().__init__(sample_rate=16000, params=VADParams(stop_secs=0.2))
        self._states = [VADState.SPEAKING, VADState.QUIET]

    def set_params(self, params):
        super().set_params(params)
        self._vad_stop_frames += EXTRA_STOP_BLOCKS

    def num_frames_required(self) -> int:
        return 512

    def voice_confidence(self, buffer: bytes) -> float:
        return 0.9

    async def analyze_audio(self, buffer: bytes) -> VADState:
        return self._states.pop(0) if self._states else VADState.QUIET


def _audio():
    return InputAudioRawFrame(audio=b"\x00" * 1024, sample_rate=16000, num_channels=1)


def _stop_frames(frames):
    return [f for f in frames if isinstance(f, VADUserStoppedSpeakingFrame)]


@pytest.mark.asyncio
async def test_user_aggregator_stamps_the_counted_debounce_on_the_stop_frame():
    vad = _BumpedVAD()
    aggregator = LLMUserAggregator(LLMContext(), params=LLMUserAggregatorParams(vad_analyzer=vad))

    received_down, _ = await run_test(aggregator, frames_to_send=[_audio(), _audio()])

    stops = _stop_frames(received_down)
    assert len(stops) == 1, [type(f).__name__ for f in received_down]
    assert stops[0].stop_secs == 0.2
    assert stops[0].debounce_secs == pytest.approx(vad.stop_debounce_secs)
    assert stops[0].debounce_secs > 0.5  # 0.192 counted + 0.384 extra


@pytest.mark.asyncio
async def test_vad_processor_stamps_the_counted_debounce_on_the_stop_frame():
    vad = _BumpedVAD()
    processor = VADProcessor(vad_analyzer=vad)

    received_down, _ = await run_test(processor, frames_to_send=[_audio(), _audio()])

    stops = _stop_frames(received_down)
    assert len(stops) == 1, [type(f).__name__ for f in received_down]
    assert stops[0].stop_secs == 0.2
    assert stops[0].debounce_secs == pytest.approx(vad.stop_debounce_secs)
