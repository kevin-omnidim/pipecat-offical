#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fork-local regression: a turn-start broadcast echo must not wedge the turn.

Kept in its own file so upstream's ``test_user_turn_controller.py`` can be
taken wholesale on the next upgrade.

The failure this pins (production call 1780458849.5219, 2026-06-03): a late
final transcript re-opened a user turn through a transcription start strategy,
the synthetic ``UserStartedSpeakingFrame`` that accompanies a turn start echoed
back into the controller, and ``_user_speaking`` latched true with no speech
behind it and nothing to clear it. Both exits from a turn consult that flag —
``_trigger_user_turn_stop`` refuses to finalize while it is set, and the
stop-timeout watchdog refuses to fire — so the turn stayed open and the bot
waited on a caller who had stopped talking. It took 24 seconds and a re-prompt
("Hello, are you here?") to recover.

Where those frames are the authoritative speech signal (external strategies,
e.g. an STT that owns turn detection) they must keep driving the flag; that
direction is pinned here too.
"""

import asyncio
import unittest

from pipecat.frames.frames import (
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop import BaseUserTurnStopStrategy
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies, UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager

USER_TURN_STOP_TIMEOUT = 0.3


class _VADGatedStopStrategy(BaseUserTurnStopStrategy):
    """Stand-in for the production stop-strategy family (TurnAnalyzer-led): the
    stop can only be committed off VAD evidence, so a turn opened with no VAD
    activity behind it gives this strategy nothing to react to.
    """

    async def process_frame(self, frame):
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            await self.trigger_user_turn_inference_triggered()
            await self.trigger_user_turn_stopped()
            return ProcessFrameResult.STOP
        return ProcessFrameResult.CONTINUE


class TestUserTurnDeadlock(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_manager = TaskManager(loop=asyncio.get_running_loop())

    def _controller(self, strategies_cls=UserTurnStrategies):
        return UserTurnController(
            user_turn_strategies=strategies_cls(
                start=[MinWordsUserTurnStartStrategy(min_words=3)],
                stop=[_VADGatedStopStrategy()],
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )

    async def test_phantom_turn_start_echo_cannot_defeat_stop_timeout(self):
        controller = self._controller()
        await controller.setup(self.task_manager)

        stops = []
        timeouts = []

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            stops.append(strategy)

        @controller.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(controller):
            timeouts.append(True)

        # A genuine turn: VAD start, transcript, VAD stop. The VAD-gated
        # strategy closes it.
        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(
            TranscriptionFrame(text="okay let's finalize", user_id="", timestamp="")
        )
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        self.assertEqual(len(stops), 1, "the genuine turn should close")

        # A late final token re-opens the turn (the min-words threshold drops
        # while the bot isn't speaking)...
        await controller.process_frame(
            TranscriptionFrame(text="finalize", user_id="", timestamp="")
        )
        self.assertTrue(controller._user_turn, "phantom start should open the turn")

        # ...and the synthetic broadcast echoes back. The caller is silent, so
        # no VAD frame follows and the VAD-gated strategy can never fire.
        await controller.process_frame(UserStartedSpeakingFrame())

        # The watchdog is the only way out.
        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.15)
        self.assertTrue(
            timeouts,
            "the stop-timeout watchdog must still fire for a stuck-open turn",
        )
        self.assertEqual(len(stops), 2, "the stuck turn must be force-stopped")
        self.assertFalse(controller._user_turn, "turn must be closed")

    async def test_external_strategies_keep_speaking_frames_authoritative(self):
        """The other direction: where these frames ARE the speech signal, they
        must still suppress the watchdog while the user is talking."""
        controller = self._controller(ExternalUserTurnStrategies)
        await controller.setup(self.task_manager)

        timeouts = []

        @controller.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(controller):
            timeouts.append(True)

        await controller.process_frame(
            TranscriptionFrame(text="still talking here", user_id="", timestamp="")
        )
        await controller.process_frame(UserStartedSpeakingFrame())
        self.assertTrue(controller._user_speaking)

        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.15)
        self.assertFalse(
            timeouts, "the watchdog must not fire while the user is genuinely speaking"
        )


if __name__ == "__main__":
    unittest.main()
