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
    InterruptionFrame,
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


#
# An interruption is not a transcript. It reaches the STT on every user turn
# start and on any component that cuts the bot off, so neither stopping the
# armed clock (it reports "speech end -> whatever interrupted") nor cancelling
# it (the utterance's real final then reports nothing) is a measurement. The
# clock is left alone; a VAD start disarms it and the no-show window bounds it.
#


@pytest.mark.asyncio
async def test_interruption_leaves_the_armed_clock_for_the_real_final():
    # Window far larger than the test: only a transcript can end the clock.
    stt = _ClockProbeSTT(stt_ttfb_timeout=5.0, ttfs_p99_latency=0.1)

    received = await _run_stt_test(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(stop_secs=0.05),
            SleepFrame(0.1),
            InterruptionFrame(),
            SleepFrame(0.1),
            _final("the utterance's real transcript"),
        ],
    )

    values = _ttfb_values(received)
    assert len(values) == 1, f"the final after an interruption must report, got {values}"
    # Measured to the final (~0.2 s after arming), not to the interruption.
    assert 0.15 < values[0] < 1.0, values


@pytest.mark.asyncio
async def test_interruption_with_no_transcript_reports_nothing():
    stt = _ClockProbeSTT(stt_ttfb_timeout=0.2, ttfs_p99_latency=0.1)

    received = await _run_stt_test(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(stop_secs=0.05),
            SleepFrame(0.05),
            InterruptionFrame(),
            # Let the no-show window expire with no transcript at all.
            SleepFrame(0.5),
        ],
    )

    assert _ttfb_values(received) == []


#
# A final that lands BEFORE the VAD stop (Soniox's server endpoint beats our
# 0.6–1.0 s silence debounce on most turns) is that utterance's transcript.
# The stop must report speech-end → that final at once, not arm a clock that
# no later transcript will stop.
#


@pytest.mark.asyncio
async def test_final_before_vad_stop_is_reported_at_the_stop():
    # Window far larger than the test: a report inside it can only come from
    # the stop handler itself, never from the no-show reporter.
    stt = _ClockProbeSTT(stt_ttfb_timeout=5.0, ttfs_p99_latency=0.1)

    received = await _run_stt_test(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            SleepFrame(0.1),
            _final("i cannot buy your flat"),
            # VAD declares the stop 0.1 s later, after 0.3 s of debounce: the
            # speech end it implies (stop − 0.3) is 0.2 s BEFORE the final.
            SleepFrame(0.1),
            VADUserStoppedSpeakingFrame(stop_secs=0.3),
            SleepFrame(0.1),
        ],
    )

    values = _ttfb_values(received)
    # One report, and it arrived inside a 5 s window that never expired: only
    # the stop handler can have produced it. Frames carry construction-time
    # timestamps while the final's arrival is measured when it is processed,
    # so the value is "test wall time so far", not a fixed 0.2 s — pin the
    # magnitude only.
    assert len(values) == 1, f"expected the final to be reported once, got {values}"
    assert 0 < values[0] < 1.0, values


#
# The mirror case: a final that lands BEFORE the utterance's speech end (an
# endpoint fired during a pause, then silence) is not this stop's transcript.
# The no-show reporter must discard the clock, not report a negative TTFB.
#


def _all_ttfb_values(frames):
    return [
        data.value
        for frame in frames
        if isinstance(frame, MetricsFrame)
        for data in frame.data
        if isinstance(data, TTFBMetricsData)
    ]


@pytest.mark.asyncio
async def test_expiry_with_only_a_final_older_than_speech_end_reports_nothing():
    stt = _ClockProbeSTT(stt_ttfb_timeout=0.2, ttfs_p99_latency=0.1)

    received = await _run_stt_test(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            _final("hello"),
            # VAD frames are system frames and overtake queued data frames in
            # the harness; let the final land first, as it does in production
            # where the STT pushes its own transcripts inline.
            SleepFrame(0.05),
            # A stop whose implied speech end (timestamp − stop_secs) is far
            # AFTER that final: the final cannot belong to this speech end.
            VADUserStoppedSpeakingFrame(stop_secs=0.05, timestamp=time.time() + 30.0),
            # Let the no-show window expire.
            SleepFrame(0.4),
        ],
    )

    negatives = [v for v in _all_ttfb_values(received) if v < 0]
    assert negatives == [], f"a transcript older than speech end is not a TTFB: {negatives}"
    assert _ttfb_values(received) == []


#
# Speech end comes from the silence the VAD really counted. A runtime debounce
# change (our Adaptive Silence) moves the counter but leaves the reported
# ``stop_secs`` alone, so a clock started from ``stop_secs`` under-measures by
# the whole extra debounce (prod 2026-09-16: 400–800 ms low on bumped turns).
#


@pytest.mark.asyncio
async def test_speech_end_uses_the_counted_debounce_when_the_frame_carries_it():
    stt = _ClockProbeSTT(stt_ttfb_timeout=5.0, ttfs_p99_latency=0.1)
    now = time.time()

    received = await _run_stt_test(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            # Reported 0.05 s, but 0.65 s of silence was actually counted, so
            # speech ended 0.65 s before the stop. The final lands right at the
            # stop (an explicit timestamp keeps the arithmetic exact).
            VADUserStoppedSpeakingFrame(stop_secs=0.05, debounce_secs=0.65, timestamp=now),
            _final("bumped turn"),
        ],
    )

    values = _ttfb_values(received)
    assert len(values) == 1, values
    # From stop_secs alone this would read ~0.05 s + processing; from the
    # counted debounce it is ~0.65 s + processing.
    assert values[0] > 0.6, values


@pytest.mark.asyncio
async def test_speech_end_falls_back_to_stop_secs_when_debounce_is_unknown():
    stt = _ClockProbeSTT(stt_ttfb_timeout=5.0, ttfs_p99_latency=0.1)
    now = time.time()

    received = await _run_stt_test(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            # A producer that does not know the counted debounce leaves it 0.
            VADUserStoppedSpeakingFrame(stop_secs=0.05, timestamp=now),
            _final("plain turn"),
        ],
    )

    values = _ttfb_values(received)
    assert len(values) == 1, values
    assert values[0] < 0.6, values


#
# A counted debounce of zero is a measurement, not a missing value: an
# analyzer whose own silence hold lives below pipecat (the AIC VAD) counts no
# stop frames, so its speech end IS the stop timestamp.
#


@pytest.mark.asyncio
async def test_a_counted_zero_debounce_arms_the_clock_and_measures_from_the_stop():
    stt = _ClockProbeSTT(stt_ttfb_timeout=5.0, ttfs_p99_latency=0.1)

    received = await _run_stt_test(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            # Reported stop_secs is 0.0 too, so only the counted value can arm this.
            VADUserStoppedSpeakingFrame(stop_secs=0.0, debounce_secs=0.0, timestamp=time.time()),
            _final("counted zero"),
        ],
    )

    values = _ttfb_values(received)
    assert len(values) == 1, f"a counted zero must still be measured, got {values}"
    assert values[0] < 0.5, values


@pytest.mark.asyncio
async def test_an_unknown_debounce_with_no_reported_stop_secs_does_not_arm():
    stt = _ClockProbeSTT(stt_ttfb_timeout=0.2, ttfs_p99_latency=0.1)

    received = await _run_stt_test(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            # A producer that reports neither value places no speech end.
            VADUserStoppedSpeakingFrame(stop_secs=0.0),
            _final("nothing to measure from"),
            SleepFrame(0.4),
        ],
    )

    assert _ttfb_values(received) == []


#
# One utterance, one clock: a second stop with no start between re-arms, and a
# final already reported belongs to no later stop.
#


@pytest.mark.asyncio
async def test_a_second_stop_without_a_start_takes_over_the_earlier_stop_s_timer():
    # Window = max(0.2, 0.1 * 1.5) = 0.2 s. The final lands past the FIRST
    # stop's window and inside the second's, so it survives only if the first
    # stop's timer no longer owns the clock.
    stt = _ClockProbeSTT(stt_ttfb_timeout=0.2, ttfs_p99_latency=0.1)

    received = await _run_stt_test(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(stop_secs=0.05),
            SleepFrame(0.15),
            VADUserStoppedSpeakingFrame(stop_secs=0.05),
            SleepFrame(0.15),
            _final("late but inside the second window"),
        ],
    )

    values = _ttfb_values(received)
    assert len(values) == 1, f"the second stop's clock must survive, got {values}"


@pytest.mark.asyncio
async def test_a_final_already_reported_is_not_reported_again_by_a_later_stop():
    stt = _ClockProbeSTT(stt_ttfb_timeout=5.0, ttfs_p99_latency=0.1)
    now = time.time()

    received = await _run_stt_test(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            _final("only utterance"),
            SleepFrame(0.05),
            # Reports the final above.
            VADUserStoppedSpeakingFrame(stop_secs=0.3, timestamp=now),
            SleepFrame(0.05),
            # A later stop whose debounce window still spans that final.
            VADUserStoppedSpeakingFrame(stop_secs=0.5, timestamp=now + 0.1),
            SleepFrame(0.1),
        ],
    )

    values = _ttfb_values(received)
    assert len(values) == 1, f"a consumed final is not a second measurement: {values}"


@pytest.mark.asyncio
async def test_a_final_that_closed_a_clock_is_not_reported_again_by_a_later_stop():
    # The final closes the measurement armed by the first stop. A second stop
    # with no speech start between must not find it still spendable.
    stt = _ClockProbeSTT(stt_ttfb_timeout=5.0, ttfs_p99_latency=0.1)
    now = time.time()

    received = await _run_stt_test(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=now),
            SleepFrame(0.05),
            _final("closes the first stop's clock"),
            SleepFrame(0.05),
            # Its debounce window reaches back past that final.
            VADUserStoppedSpeakingFrame(stop_secs=1.0, timestamp=now + 0.2),
            SleepFrame(0.1),
        ],
    )

    values = _ttfb_values(received)
    assert len(values) == 1, f"the final closed one measurement, not two: {values}"
