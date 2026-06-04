#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""STT TTFB clock correctness.

An STT TTFB measures user speech-end → first finalized transcript **of that
same utterance**. A clock armed by one utterance must never be stopped by
another utterance's transcript. These tests pin the two production leaks that
created phantom TTFB values (6–20 s reported on sub-second transcriptions):

- Leak 1: the no-show timeout expiring with zero transcripts left the clock
  armed (it reported nothing but never disarmed).
- Leak 2: a new VAD speech-start cancelled only the timeout *task*, leaving
  the previous utterance's armed clock to be consumed by the next finalized
  transcript.

They also pin the per-provider no-show window (a flat window shorter than a
provider's TTFS p99 silently swallows slow-but-real finals) and the preserved
report-at-arrival behavior for the timeout reporter.
"""

import time

import pytest

from pipecat.frames.frames import (
    MetricsFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.pipeline.task import PipelineParams
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics
from pipecat.services.stt_service import STTService
from pipecat.tests.utils import SleepFrame, run_test


class _ClockProbeSTT(STTService):
    """Minimal STT service; tests inject transcripts as pipeline frames."""

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio):
        return
        yield  # pragma: no cover


def _ttfb_values(frames):
    values = []
    for frame in frames:
        if isinstance(frame, MetricsFrame):
            for data in frame.data:
                # Skip the zeroed initial MetricsFrame pipecat emits on
                # StartFrame — only actual measurements matter here, and every
                # scenario below arms the clock at least stop_secs in the past
                # so a real measurement is always > 0.
                if isinstance(data, TTFBMetricsData) and data.value > 0:
                    values.append(data.value)
    return values


def _final(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="", timestamp="", finalized=True)


def _interim(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="", timestamp="", finalized=False)


async def _run_stt_test(stt, frames_to_send):
    received_down, _ = await run_test(
        stt,
        frames_to_send=frames_to_send,
        pipeline_params=PipelineParams(enable_metrics=True),
    )
    return received_down


#
# Metrics-level: the cancel helper.
#


@pytest.mark.asyncio
async def test_cancel_ttfb_metrics_discards_armed_clock():
    metrics = FrameProcessorMetrics()
    metrics.set_processor_name("STTService#0")

    await metrics.start_ttfb_metrics(start_time=time.time() - 5.0, report_only_initial_ttfb=False)
    await metrics.cancel_ttfb_metrics()

    assert await metrics.stop_ttfb_metrics() is None


@pytest.mark.asyncio
async def test_cancel_ttfb_metrics_then_rearm_reports_only_new_measurement():
    metrics = FrameProcessorMetrics()
    metrics.set_processor_name("STTService#0")

    # A stale armed clock from 5 s ago is discarded...
    await metrics.start_ttfb_metrics(start_time=time.time() - 5.0, report_only_initial_ttfb=False)
    await metrics.cancel_ttfb_metrics()

    # ...and a fresh measurement is unaffected by it.
    start = time.time()
    await metrics.start_ttfb_metrics(start_time=start, report_only_initial_ttfb=False)
    frame = await metrics.stop_ttfb_metrics(end_time=start + 0.5)

    assert frame is not None
    assert frame.data[0].value == pytest.approx(0.5, abs=0.01)


@pytest.mark.asyncio
async def test_cancel_ttfb_metrics_is_a_noop_when_nothing_armed():
    metrics = FrameProcessorMetrics()
    metrics.set_processor_name("STTService#0")

    await metrics.cancel_ttfb_metrics()

    assert await metrics.stop_ttfb_metrics() is None


#
# Leak 1: no-show expiry with zero transcripts must disarm the clock.
#


@pytest.mark.asyncio
async def test_vad_false_start_never_produces_phantom_ttfb():
    # Window = max(0.2, 0.1 * 1.5) = 0.2 s.
    stt = _ClockProbeSTT(stt_ttfb_timeout=0.2, ttfs_p99_latency=0.1)

    received = await _run_stt_test(
        stt,
        [
            # Utterance 1: a VAD false-start — no speech, no transcript, ever.
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(stop_secs=0.05),
            # The no-show window expires with zero transcripts.
            SleepFrame(0.5),
            # Utterance 2: a real one. Its first finalized transcript arrives
            # mid-utterance — in production this is the frame that consumed
            # the stale clock and reported the phantom.
            VADUserStartedSpeakingFrame(),
            _final("yes"),
            SleepFrame(0.05),
            VADUserStoppedSpeakingFrame(stop_secs=0.05),
            _final("yes my pin code is"),
        ],
    )

    values = _ttfb_values(received)
    # Exactly one TTFB: utterance 2's. The false-start contributes nothing.
    assert len(values) == 1, f"expected 1 TTFB report, got {values}"


#
# Leak 2: a new VAD speech-start must disarm a leftover clock on its own,
# even when the no-show window has NOT yet expired.
#


@pytest.mark.asyncio
async def test_new_vad_start_clears_leftover_armed_clock():
    # Window far larger than the gap between utterances: expiry never fires
    # inside this test, so only the VAD-start clearing can prevent a phantom.
    stt = _ClockProbeSTT(stt_ttfb_timeout=5.0, ttfs_p99_latency=0.1)

    received = await _run_stt_test(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(stop_secs=0.05),
            SleepFrame(0.3),
            VADUserStartedSpeakingFrame(),
            _final("hello"),
            SleepFrame(0.05),
            VADUserStoppedSpeakingFrame(stop_secs=0.05),
            _final("hello are you there"),
        ],
    )

    values = _ttfb_values(received)
    assert len(values) == 1, f"expected 1 TTFB report, got {values}"


#
# Per-provider no-show window: a finalized transcript slower than the flat
# timeout but within the provider's window must still be reported.
#


def test_no_show_window_derives_from_provider_ttfs_p99():
    azure_like = _ClockProbeSTT(stt_ttfb_timeout=0.2, ttfs_p99_latency=0.3)
    soniox_like = _ClockProbeSTT(stt_ttfb_timeout=0.2, ttfs_p99_latency=0.05)
    unknown = _ClockProbeSTT(stt_ttfb_timeout=0.2, ttfs_p99_latency=None)

    assert azure_like.stt_ttfb_no_show_window == pytest.approx(0.45)
    assert soniox_like.stt_ttfb_no_show_window == pytest.approx(0.2)
    assert unknown.stt_ttfb_no_show_window == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_late_final_within_provider_window_is_still_reported():
    # Flat timeout 0.2 s, provider p99 0.3 s → window 0.45 s. The final lands
    # at ~0.3 s: past the flat timeout, inside the window. A naive
    # cancel-at-flat-timeout implementation would swallow it.
    stt = _ClockProbeSTT(stt_ttfb_timeout=0.2, ttfs_p99_latency=0.3)

    received = await _run_stt_test(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(stop_secs=0.05),
            SleepFrame(0.3),
            _final("slow but real"),
        ],
    )

    values = _ttfb_values(received)
    assert len(values) == 1, f"expected 1 TTFB report, got {values}"
    # Armed at (VAD-stop timestamp − stop_secs); the final arrived ~0.3 s
    # later. Generous tolerance: only the order of magnitude matters — the
    # value must be the real wait, not zero and not a multi-second phantom.
    assert 0.2 < values[0] < 1.0


#
# Timeout reporter (non-finalizing providers): expiry WITH a transcript seen
# still reports, measured to the transcript's arrival time — preserved
# behavior, pinned so the expiry changes cannot regress it.
#


@pytest.mark.asyncio
async def test_expiry_with_interim_transcript_reports_at_arrival_time():
    stt = _ClockProbeSTT(stt_ttfb_timeout=0.2, ttfs_p99_latency=0.1)

    received = await _run_stt_test(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(stop_secs=0.05),
            SleepFrame(0.1),
            _interim("partial words"),
            # Let the no-show window expire; the timeout reporter fires using
            # the interim's arrival time as the end time.
            SleepFrame(0.4),
        ],
    )

    values = _ttfb_values(received)
    assert len(values) == 1, f"expected 1 TTFB report, got {values}"
    # End time is the interim's arrival (~0.1–0.15 s after arming), NOT the
    # expiry moment (~0.45 s) — pin that distinction with room for jitter.
    assert values[0] < 0.35


#
# Back-to-back utterances with no false-start: two clean reports — the fix
# must not suppress legitimate consecutive measurements.
#


@pytest.mark.asyncio
async def test_two_clean_utterances_report_two_fresh_ttfbs():
    stt = _ClockProbeSTT(stt_ttfb_timeout=0.2, ttfs_p99_latency=0.1)

    received = await _run_stt_test(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(stop_secs=0.05),
            _final("first utterance"),
            SleepFrame(0.3),
            VADUserStartedSpeakingFrame(),
            SleepFrame(0.05),
            VADUserStoppedSpeakingFrame(stop_secs=0.05),
            _final("second utterance"),
        ],
    )

    values = _ttfb_values(received)
    assert len(values) == 2, f"expected 2 TTFB reports, got {values}"
