#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (
    MinWordsUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    BaseUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
    deferred,
)
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies, UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

USER_TURN_STOP_TIMEOUT = 0.2
TRANSCRIPTION_TIMEOUT = 0.1


class TestUserTurnController(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def test_default_user_turn_strategies(self):
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)],
            )
        )

        await controller.setup(self.task_manager)

        should_start = None
        should_stop = None

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal should_start
            should_start = True

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal should_stop
            should_stop = True

        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertTrue(should_start)
        self.assertFalse(should_stop)

        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )
        self.assertTrue(should_start)
        self.assertFalse(should_stop)

        await controller.process_frame(VADUserStoppedSpeakingFrame())
        self.assertTrue(should_start)
        # Wait for user_speech_timeout to elapse
        await asyncio.sleep(TRANSCRIPTION_TIMEOUT + 0.1)
        self.assertTrue(should_stop)

    async def test_inference_triggered_fires_alongside_stopped(self):
        """Default strategies fire both inference-triggered and stopped, in order."""
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)],
            )
        )

        await controller.setup(self.task_manager)

        events: list[str] = []

        @controller.event_handler("on_user_turn_inference_triggered")
        async def on_user_turn_inference_triggered(controller, strategy):
            events.append("inference_triggered")

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            events.append("stopped")

        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(TRANSCRIPTION_TIMEOUT + 0.1)

        self.assertEqual(events, ["inference_triggered", "stopped"])

    async def test_deferred_wrapper_skips_stopped(self):
        """A deferred() wrapper drops the inner strategy's on_user_turn_stopped event."""
        wrapped = deferred(
            SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)
        )
        controller = UserTurnController(user_turn_strategies=UserTurnStrategies(stop=[wrapped]))

        await controller.setup(self.task_manager)

        events: list[str] = []

        @controller.event_handler("on_user_turn_inference_triggered")
        async def on_user_turn_inference_triggered(controller, strategy):
            events.append("inference_triggered")

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            events.append("stopped")

        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(TRANSCRIPTION_TIMEOUT + 0.1)

        # The inner strategy fires inference-triggered (forwarded by the
        # wrapper). Finalization is suppressed, but the controller's
        # stop watchdog eventually fires `stopped`.
        self.assertEqual(events[0], "inference_triggered")

    async def test_user_turn_start_reset(self):
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=3)]
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )

        await controller.setup(self.task_manager)

        should_start = 0

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal should_start
            should_start += 1

        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(TranscriptionFrame(text="One", user_id="cat", timestamp=""))
        self.assertEqual(should_start, 0)

        await controller.process_frame(
            TranscriptionFrame(text="One two three!", user_id="cat", timestamp="")
        )
        self.assertEqual(should_start, 1)

        # Trigger user stop turn so we can trigger user start turn again.
        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.1)

        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(TranscriptionFrame(text="Hi!", user_id="cat", timestamp=""))
        self.assertEqual(should_start, 1)

        await controller.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertEqual(should_start, 2)

    async def test_user_turn_stop_timeout_no_transcription(self):
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )

        await controller.setup(self.task_manager)

        should_start = None
        should_stop = None
        timeout = None

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal should_start
            should_start = True

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal should_stop
            should_stop = True

        @controller.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(controller):
            nonlocal timeout
            timeout = True

        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertTrue(should_start)
        self.assertFalse(should_stop)
        self.assertFalse(timeout)

        await controller.process_frame(VADUserStoppedSpeakingFrame())
        self.assertTrue(should_start)
        self.assertFalse(should_stop)

        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.1)
        self.assertTrue(should_start)
        self.assertTrue(should_stop)
        self.assertTrue(timeout)

    async def test_external_user_turn_strategies_no_timeout_while_speaking(self):
        """Test that timeout does not trigger when user is still speaking with external strategies."""
        controller = UserTurnController(
            user_turn_strategies=ExternalUserTurnStrategies(),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )

        await controller.setup(self.task_manager)

        should_start = None
        should_stop = None
        timeout = None

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal should_start
            should_start = True

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal should_stop
            should_stop = True

        @controller.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(controller):
            nonlocal timeout
            timeout = True

        # Simulate external service (like Deepgram Flux) broadcasting UserStartedSpeakingFrame
        await controller.process_frame(UserStartedSpeakingFrame())
        self.assertTrue(should_start)
        self.assertFalse(should_stop)
        self.assertFalse(timeout)

        # User is still speaking, timeout should not trigger
        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.1)
        self.assertTrue(should_start)
        self.assertFalse(should_stop)
        self.assertFalse(timeout)

        # Now external service broadcasts UserStoppedSpeakingFrame
        await controller.process_frame(UserStoppedSpeakingFrame())

        # But no transcription, so timeout should trigger
        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.1)

        self.assertTrue(should_start)
        self.assertTrue(should_stop)
        self.assertTrue(timeout)

    async def test_late_transcription_between_turns_no_premature_stop(self):
        """Test that a late transcription arriving between turns does not cause a premature stop.

        Reproduces the bug from issue #4053: after turn 1 completes and reset()
        clears state, a late TranscriptionFrame sets _text to stale content. On
        the next turn, that stale _text gates a premature turn stop via timeout(0)
        before the current turn's transcript arrives.

        Uses only VADUserTurnStartStrategy (no TranscriptionUserTurnStartStrategy)
        so the late transcription doesn't trigger a spurious turn start.
        """
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)],
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )

        await controller.setup(self.task_manager)

        start_count = 0
        stop_count = 0

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal start_count
            start_count += 1

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal stop_count
            stop_count += 1

        # === Turn 1: S-T-E ===
        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(start_count, 1)

        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )

        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(TRANSCRIPTION_TIMEOUT + 0.1)
        self.assertEqual(stop_count, 1)

        # === Between turns: late transcription arrives ===
        # This sets _text on the stop strategy while _user_turn is False.
        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )

        # === Turn 2: S-T-E (transcription arrives during turn) ===
        # The fix resets stop strategies at turn start, clearing stale _text.
        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(start_count, 2)

        await controller.process_frame(
            TranscriptionFrame(text="How are you?", user_id="", timestamp="now")
        )

        await controller.process_frame(VADUserStoppedSpeakingFrame())

        # Wait for user_speech_timeout to elapse — should get turn 2 stop
        await asyncio.sleep(TRANSCRIPTION_TIMEOUT + 0.1)
        self.assertEqual(stop_count, 2)


if __name__ == "__main__":
    unittest.main()


class _VADGatedStopStrategy(BaseUserTurnStopStrategy):
    """Minimal stand-in for the production stop-strategy family
    (TurnAnalyzer-led): the turn stop can ONLY be committed off VAD
    evidence. A turn opened without any VAD activity behind it gives
    this strategy nothing to react to — exactly the production
    configuration in the 2026-06-03 deadlock.
    """

    async def reset(self):
        pass

    async def process_frame(self, frame):
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            await self.trigger_user_turn_inference_triggered()
            await self.trigger_user_turn_stopped()
            return ProcessFrameResult.STOP
        return ProcessFrameResult.CONTINUE


class TestUserTurnDeadlock(unittest.IsolatedAsyncioTestCase):
    """Regression for the 2026-06-03 turn-deadlock (call 1780458849.5219):
    a late final transcript re-opened the user turn via a transcription
    start strategy (MinWords drops to 1 word when the bot isn't
    speaking), the aggregator's synthetic UserStartedSpeakingFrame
    echoed back into the controller and set ``_user_speaking``
    stale-true, and the stop-timeout watchdog — guarded by
    ``not _user_speaking`` — never fired. With VAD-gated stop
    strategies (production: TurnAnalyzer-led) no strategy can close a
    turn that has no VAD activity behind it, so the conversation
    starved until the caller re-prompted ("Hello, are you here?") 24
    seconds later.

    With standard (VAD/transcription) strategies the synthetic
    UserStarted/StoppedSpeakingFrames are the controller's own turn
    byproducts, not speech evidence — only VAD frames are. External
    strategies (where those frames ARE the authoritative signal) keep
    the existing behavior, pinned by
    ``test_external_user_turn_strategies_no_timeout_while_speaking``.
    """

    async def asyncSetUp(self):
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    def _make_controller(self):
        return UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=3)],
                stop=[_VADGatedStopStrategy()],
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )

    async def test_phantom_turn_start_echo_cannot_defeat_stop_timeout(self):
        controller = self._make_controller()
        await controller.setup(self.task_manager)

        stops = []
        timeouts = []

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            stops.append(strategy)

        @controller.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(controller):
            timeouts.append(True)

        # The genuine turn: VAD start, transcript, VAD stop — the
        # VAD-gated strategy closes it.
        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(
            TranscriptionFrame(text="okay let's finalize", user_id="", timestamp="")
        )
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        self.assertEqual(len(stops), 1, "the genuine turn should close")

        # The phantom: a LATE final token re-opens the turn (MinWords
        # threshold is 1 while the bot isn't speaking)…
        await controller.process_frame(
            TranscriptionFrame(text="finalize", user_id="", timestamp="")
        )
        self.assertTrue(controller._user_turn, "phantom start should open the turn")
        # …and the aggregator's synthetic broadcast echoes back into the
        # controller. The user is silent; no VAD frame accompanies it,
        # so the VAD-gated stop strategy can never fire.
        await controller.process_frame(UserStartedSpeakingFrame())

        # Nothing else ever arrives. The watchdog is the only way out.
        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.15)
        self.assertTrue(
            timeouts,
            "stop-timeout watchdog must fire for a stuck-open turn even "
            "when the synthetic UserStartedSpeakingFrame echo set "
            "_user_speaking stale-true",
        )
        self.assertEqual(len(stops), 2, "the stuck turn must be force-stopped")
        self.assertFalse(controller._user_turn, "turn must be closed")
