#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Smallest AI text-to-speech service implementation.

This module provides a WebSocket-based integration with Smallest AI's Waves
API for real-time text-to-speech synthesis.
"""

import asyncio
import base64
import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat import version as pipecat_version
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven, is_given
from pipecat.services.tts_service import AudioContextWordTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Smallest AI, you need to `pip install pipecat-ai[smallest]`.")
    raise Exception(f"Missing module: {e}")


_MODEL_DEFAULT_VOICES: Dict[str, str] = {
    "lightning_v3.1": "sophia",
    "lightning_v3.1_pro": "meher",
}

# Voices that emit word-timestamp events on the base queue. Other voices
# accept word_timestamps=True without error but simply emit no word events.
_WORD_TIMESTAMP_VOICES = {"meher", "devansh", "kartik", "maithili", "liam", "avery"}


def language_to_smallest_tts_language(language: Language) -> str:
    """Convert a Language enum to a Smallest TTS language string.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Smallest language code, falling back to the base
        code (e.g. ``en`` from ``en-US``) when not in the verified mapping.
    """
    LANGUAGE_MAP = {
        Language.AR: "ar",
        Language.BN: "bn",
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es",
        Language.FR: "fr",
        Language.GU: "gu",
        Language.HE: "he",
        Language.HI: "hi",
        Language.IT: "it",
        Language.KN: "kn",
        Language.MR: "mr",
        Language.NL: "nl",
        Language.PL: "pl",
        Language.RU: "ru",
        Language.TA: "ta",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


@dataclass
class SmallestTTSSettings(TTSSettings):
    """Settings for SmallestTTSService.

    Parameters:
        speed: Speech speed multiplier (0.5-2.0).
    """

    speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class SmallestTTSService(AudioContextWordTTSService):
    """Smallest AI real-time text-to-speech service using WebSocket streaming.

    Streams synthesized audio over a WebSocket connection to Smallest AI's
    Waves API, correlating audio and (optionally) word timestamps with the
    active turn via Pipecat's audio-context mechanism.
    """

    _settings: SmallestTTSSettings

    class InputParams(BaseModel):
        """Configuration parameters for Smallest AI TTS service.

        Parameters:
            language: Language for synthesis. Defaults to English.
            speed: Speech speed multiplier (0.5-2.0).
        """

        language: Optional[Language] = Language.EN
        speed: Optional[float] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice: Optional[str] = None,
        model: str = "lightning_v3.1_pro",
        base_url: str = "wss://api.smallest.ai",
        sample_rate: Optional[int] = None,
        output_format: str = "pcm",
        word_timestamps: bool = True,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Smallest AI WebSocket TTS service.

        Args:
            api_key: Smallest AI API key for authentication.
            voice: Voice identifier. Defaults to the model's default voice.
            model: Model identifier: ``lightning_v3.1`` or ``lightning_v3.1_pro``.
            base_url: Base WebSocket URL for the Smallest API.
            sample_rate: Output audio sample rate in Hz. If None, uses the
                pipeline's configured sample rate.
            output_format: Audio format returned by the API. Fixed at init time.
            word_timestamps: Whether to request per-word timing events. Supported
                on base-queue English/Hindi voices; other voices emit no word
                events but function normally, so leaving this on is safe.
            params: Additional configuration parameters.
            **kwargs: Additional arguments passed to parent AudioContextWordTTSService.
        """
        super().__init__(
            push_text_frames=not word_timestamps,
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or SmallestTTSService.InputParams()
        resolved_voice = voice or _MODEL_DEFAULT_VOICES.get(model, "meher")

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._output_format = output_format
        self._word_timestamps = word_timestamps

        self._settings = SmallestTTSSettings(
            model=model,
            voice=resolved_voice,
            language=self.language_to_service_language(params.language)
            if params.language
            else "en",
            speed=params.speed if params.speed is not None else NOT_GIVEN,
        )
        self._sync_model_name_to_metrics()

        self._receive_task = None
        self._keepalive_task = None

        # Smallest reports word timestamps relative to each request's own
        # audio, so multiple run_tts() calls within one turn are offset onto
        # a continuous timeline using the cumulative time of prior requests.
        # Mirrors the pattern used by RimeTTSService in this codebase.
        self._cumulative_time: float = 0.0
        self._request_end_time: float = 0.0
        self._wt_request_id: Optional[str] = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Smallest service language format."""
        return language_to_smallest_tts_language(language)

    def _build_msg(self, text: str) -> dict:
        """Build a WebSocket message for the Smallest API."""
        msg = {
            "text": text,
            "voice_id": self._settings.voice,
            "model": self._settings.model,
            "language": self._settings.language,
            "sample_rate": self.sample_rate,
            "output_format": self._output_format,
        }

        if is_given(self._settings.speed) and self._settings.speed is not None:
            msg["speed"] = self._settings.speed

        if self._word_timestamps:
            msg["word_timestamps"] = True

        return msg

    def _build_websocket_url(self) -> str:
        return f"{self._base_url}/waves/v1/tts/live"

    async def start(self, frame: StartFrame):
        """Start the Smallest TTS service."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Smallest TTS service."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Smallest TTS service."""
        await super().cancel(frame)
        await self._disconnect()

    async def _update_settings(self, update: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta. Fields take effect on the next message sent."""
        return await super()._update_settings(update)

    async def flush_audio(self, context_id: str | None = None):
        """Flush any pending audio data.

        Args:
            context_id: The specific context to flush. Smallest has no explicit
                flush/finalize message on this endpoint, so this is a no-op —
                present only to satisfy the base class's call signature.
        """
        logger.trace(f"{self}: flushing audio")

    async def _connect(self):
        """Connect to Smallest WebSocket and start receive/keepalive tasks."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def _disconnect(self):
        """Disconnect from Smallest WebSocket and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection to the Smallest API."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Smallest TTS")

            self._websocket = await websocket_connect(
                self._build_websocket_url(),
                additional_headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "X-Source": "pipecat",
                    "X-Pipecat-Version": pipecat_version(),
                },
            )

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Smallest TTS connection error: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Smallest TTS")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(
                error_msg=f"Smallest TTS error closing websocket: {e}", exception=e
            )
        finally:
            await self.remove_active_audio_context()
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _keepalive_task_handler(self):
        """Send periodic keepalive messages to prevent idle timeout."""
        KEEPALIVE_INTERVAL = 30
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL)
            await self._send_keepalive()

    async def _send_keepalive(self):
        """Send a silent message to keep the WebSocket connection alive."""
        if self._websocket and self._websocket.state is State.OPEN:
            msg = {
                "text": " ",
                "voice_id": self._settings.voice,
                "model": self._settings.model,
                "language": self._settings.language,
            }
            await self._websocket.send(json.dumps(msg))

    def _advance_word_timestamp_request(self, request_id: Optional[str]):
        """Roll the turn offset forward when word timestamps cross into a new request.

        Smallest reports word timestamps relative to each request's own audio
        and does not emit a per-request "complete", so a change in
        ``request_id`` marks the boundary between requests within a turn.
        """
        if request_id == self._wt_request_id:
            return
        if self._wt_request_id is not None:
            self._cumulative_time += self._request_end_time
            self._request_end_time = 0.0
        self._wt_request_id = request_id

    async def _receive_messages(self):
        """Receive and process messages from the Smallest WebSocket API."""
        async for message in self._get_websocket():
            msg = json.loads(message)
            status = msg.get("status")

            if status == "complete":
                await self.stop_all_metrics()
            elif status == "chunk":
                await self.stop_ttfb_metrics()
                await self.start_word_timestamps()
                context_id = self.get_active_audio_context_id()
                frame = TTSAudioRawFrame(
                    audio=base64.b64decode(msg["data"]["audio"]),
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
                await self.append_to_audio_context(context_id, frame)
            elif status == "word_timestamp":
                self._advance_word_timestamp_request(msg.get("request_id"))
                data = msg.get("data", {})
                word = data.get("word")
                start = data.get("start")
                end = data.get("end")
                if word is not None and start is not None:
                    context_id = self.get_active_audio_context_id()
                    await self.add_word_timestamps(
                        [(word, start + self._cumulative_time)], context_id=context_id
                    )
                    if end is not None:
                        self._request_end_time = max(self._request_end_time, end)
            elif status == "error":
                await self.push_frame(TTSStoppedFrame())
                await self.stop_all_metrics()
                await self.push_error(error_msg=f"Smallest TTS error: {msg.get('error', msg)}")
            else:
                logger.warning(f"{self} unknown message status: {msg}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Smallest's WebSocket streaming API.

        Args:
            text: The text to synthesize into speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio arrives via the WebSocket receive task.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                if not self.has_active_audio_context():
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame(context_id=context_id)
                    self._cumulative_time = 0.0
                    self._request_end_time = 0.0
                    self._wt_request_id = None
                    await self.create_audio_context(context_id)

                msg = self._build_msg(text=text)
                await self._get_websocket().send(json.dumps(msg))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Smallest TTS send error: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Smallest TTS error: {e}")