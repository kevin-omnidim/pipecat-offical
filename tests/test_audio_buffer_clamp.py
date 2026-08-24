#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fork-local: the gap fill must never push a track past the wall clock.

Kept in its own file so upstream's ``test_audio_buffer_processor.py`` can be
taken wholesale on the next upgrade. The clamp itself lives in
``AudioBufferProcessor._fill_buffer_silence_gap``.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import InputAudioRawFrame, StartFrame
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager


class _PassthroughResampler:
    async def resample(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
        return audio


async def _make_processor(*, buffer_size: int = 0) -> AudioBufferProcessor:
    """A mono processor recording at 16 kHz, initialised without a full pipeline."""
    processor = AudioBufferProcessor(sample_rate=16000, num_channels=1, buffer_size=buffer_size)
    processor._input_resampler = _PassthroughResampler()
    processor._output_resampler = _PassthroughResampler()

    await processor.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=TaskManager(loop=asyncio.get_event_loop()),
            pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
        )
    )
    await processor.process_frame(
        StartFrame(audio_out_sample_rate=16000), FrameDirection.DOWNSTREAM
    )
    await processor.start_recording()
    return processor


class TestBurstDeliveryClamp(unittest.IsolatedAsyncioTestCase):
    """A late-delivered audio burst must not inflate the recording past wall time.

    When frames stall in flight (network jitter, event-loop stall, browser tab
    throttling) and then arrive as a burst, the elapsed-based gap fill sees the
    stall as silence, fills it, and then the burst is appended on top — the
    same span recorded twice. The wall-clock clamp caps every fill at the true
    wall position, so a burst's overshoot is reclaimed at the next genuine
    silence and the track ends exactly on the wall clock (prod: call 6988499,
    a 78 s web call whose recording came out 90 s long).

    Reclamation is what this asserts, and it is also the limit of the fix: a
    burst is only *known* to be delivery lag after it lands, so a track can
    still be over-long between the burst and the next silence. See the
    ``_fill_buffer_silence_gap`` docstring for the measured residual.
    """

    _BYTES_PER_SECOND = 16000 * 2

    async def _send(self, processor: AudioBufferProcessor, audio: bytes):
        await processor.process_frame(
            InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

    async def test_burst_overshoot_reclaimed_at_next_silence_fill(self):
        p = await _make_processor()
        frame = b"\x01" * 640  # 20 ms @ 16 kHz mono

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            p._recording_start_time = 0.0
            p._last_user_buffer_update_time = 0.0

            # A 1-second stall, then the whole backlog arrives as one burst.
            mock_time.monotonic.return_value = 1.0
            for _ in range(50):  # 50 × 20 ms = 1 s of late audio
                await self._send(p, frame)

            # A genuine 2-second mute gap later: the fill must be clamped so
            # the track lands exactly back on the wall clock.
            mock_time.monotonic.return_value = 3.0
            await self._send(p, frame)

        expected_total = int(3.0 * self._BYTES_PER_SECOND)
        self.assertEqual(len(p._user_audio_buffer), expected_total)
        await p.cleanup()

    async def test_the_clamp_still_holds_across_a_periodic_flush(self):
        """Production sets buffer_size (30 s), so the buffer is emptied mid-call
        and the cap depends on the flushed-byte tally rather than on
        ``len(buffer)`` alone. If that tally is wrong the cap goes unbounded and
        the clamp quietly stops working after the first flush."""
        one_second = self._BYTES_PER_SECOND
        p = await _make_processor(buffer_size=one_second)  # flush every ~1 s
        frame = b"\x01" * 640  # 20 ms @ 16 kHz mono

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            p._recording_start_time = 0.0
            p._last_user_buffer_update_time = 0.0

            # 2 s of honest, on-time audio — crosses the flush threshold twice.
            for i in range(100):
                mock_time.monotonic.return_value = 0.02 * (i + 1)
                await self._send(p, frame)

            self.assertGreater(
                p._user_flushed_bytes, 0, "the periodic flush never fired")

            # Now a stall whose backlog lands as a burst, then a genuine mute.
            mock_time.monotonic.return_value = 3.0
            for _ in range(50):
                await self._send(p, frame)
            mock_time.monotonic.return_value = 5.0
            await self._send(p, frame)

        total = p._user_flushed_bytes + len(p._user_audio_buffer)
        self.assertEqual(total, int(5.0 * self._BYTES_PER_SECOND))
        await p.cleanup()


if __name__ == "__main__":
    unittest.main()
