#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Smallest AI speech-to-text service implementation.

This module provides a WebSocket-based real-time STT service using Smallest
AI's unified streaming endpoint (``/waves/v1/stt/live``). Audio is streamed
continuously and interim/final transcripts are received with low latency.
"""

import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional, Union
from urllib.parse import urlencode

from loguru import logger
from pydantic import BaseModel

from pipecat import version as pipecat_version
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven
from pipecat.services.stt_latency import SMALLEST_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Smallest AI, you need to `pip install pipecat-ai[smallest]`.")
    raise Exception(f"Missing module: {e}")


def language_to_smallest_stt_language(language: Language) -> str:
    """Convert a Language enum to a Smallest STT language code."""
    LANGUAGE_MAP = {
        Language.BG: "bg",
        Language.BN: "bn",
        Language.CS: "cs",
        Language.DA: "da",
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es",
        Language.ET: "et",
        Language.FI: "fi",
        Language.FR: "fr",
        Language.GU: "gu",
        Language.HI: "hi",
        Language.HU: "hu",
        Language.IT: "it",
        Language.KN: "kn",
        Language.LT: "lt",
        Language.LV: "lv",
        Language.ML: "ml",
        Language.MR: "mr",
        Language.MT: "mt",
        Language.NL: "nl",
        Language.OR: "or",
        Language.PA: "pa",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SK: "sk",
        Language.SV: "sv",
        Language.TA: "ta",
        Language.TE: "te",
        Language.UK: "uk",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


@dataclass
class SmallestSTTSettings(STTSettings):
    """Settings for SmallestSTTService.

    Parameters:
        word_timestamps: Include word-level timestamps.
        full_transcript: Include cumulative transcript.
        sentence_timestamps: Include sentence-level timestamps.
        redact_pii: Redact personally identifiable information.
        redact_pci: Redact payment card information.
        diarize: Enable speaker diarization.
    """

    word_timestamps: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    full_transcript: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    sentence_timestamps: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    redact_pii: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    redact_pci: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    diarize: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class SmallestSTTService(WebsocketSTTService):
    """Smallest AI real-time speech-to-text service using the Pulse WebSocket API.

    Streams audio continuously over a WebSocket connection and receives
    interim and final transcription results with low latency. Uses Pipecat's
    VAD to detect when the user stops speaking and sends a finalize message
    to flush the final transcript.
    """

    _settings: SmallestSTTSettings

    class InputParams(BaseModel):
        """Configuration parameters for Smallest AI STT service.

        Parameters:
            language: Language for transcription. Defaults to English. Accepts
                either a ``Language`` enum for a single fixed language, or a raw
                string for one of Smallest's multilingual streaming aggregators
                (``north_indic``, ``multi-south-indic``, ``multi-asian``), which
                auto-detect across a fixed regional language set. Aggregator
                strings are passed through to the API unchanged.
            word_timestamps: Include word-level timestamps.
            full_transcript: Include cumulative transcript.
            sentence_timestamps: Include sentence-level timestamps.
            redact_pii: Redact personally identifiable information.
            redact_pci: Redact payment card information.
            diarize: Enable speaker diarization.
        """

        language: Optional[Union[Language, str]] = Language.EN
        word_timestamps: bool = False
        full_transcript: bool = False
        sentence_timestamps: bool = False
        redact_pii: bool = False
        redact_pci: bool = False
        diarize: bool = False

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://api.smallest.ai",
        model: str = "pulse",
        encoding: str = "linear16",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        ttfs_p99_latency: Optional[float] = SMALLEST_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Smallest AI STT service.

        Args:
            api_key: Smallest AI API key for authentication.
            base_url: Base WebSocket URL for the Smallest API.
            model: STT model identifier. Currently only ``pulse`` is available.
            encoding: Audio encoding format. Defaults to "linear16".
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline's rate.
            params: Additional configuration parameters.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
            **kwargs: Additional arguments passed to the parent WebsocketSTTService.
        """
        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )

        params = params or SmallestSTTService.InputParams()

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._encoding = encoding

        # Aggregator codes (e.g. "north_indic") are already valid API values and
        # bypass the Language enum conversion; a Language enum is converted to
        # Smallest's ISO code as usual.
        if isinstance(params.language, str):
            resolved_language = params.language
        elif params.language:
            resolved_language = self.language_to_service_language(params.language)
        else:
            resolved_language = "en"

        self._settings = SmallestSTTSettings(
            model=model,
            language=resolved_language,
            word_timestamps=params.word_timestamps,
            full_transcript=params.full_transcript,
            sentence_timestamps=params.sentence_timestamps,
            redact_pii=params.redact_pii,
            redact_pci=params.redact_pci,
            diarize=params.diarize,
        )
        self._sync_model_name_to_metrics()

        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Smallest service language format."""
        return language_to_smallest_stt_language(language)

    async def start(self, frame: StartFrame):
        """Start the service and connect to the WebSocket."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and disconnect from the WebSocket."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and disconnect from the WebSocket."""
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, sending a finalize message on VAD stop."""
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._websocket and self._websocket.state is State.OPEN:
                try:
                    await self._websocket.send(json.dumps({"type": "finalize"}))
                except Exception as e:
                    logger.warning(f"{self} failed to send finalize: {e}")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio to the Smallest Pulse WebSocket for transcription.

        Args:
            audio: Raw PCM audio bytes.

        Yields:
            Frame: None -- transcription results arrive via WebSocket messages.
        """
        await self.start_processing_metrics()

        if not self._websocket or self._websocket.state is State.CLOSED:
            await self._connect()

        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(audio)
            except Exception as e:
                yield ErrorFrame(error=f"Smallest STT error: {e}")
                await self.stop_processing_metrics()
                return

        await self.stop_processing_metrics()
        yield None

    async def _send_keepalive(self, silence: bytes):
        """Send silent PCM audio to keep the WebSocket connection alive."""
        if self._websocket and self._websocket.state is State.OPEN:
            await self._websocket.send(silence)

    async def _update_settings(self, update: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect so the new query params take effect."""
        changed = await super()._update_settings(update)

        if changed:
            await self._disconnect()
            await self._connect()

        return changed

    async def _connect(self):
        await self._connect_websocket()
        await super()._connect()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection to the Smallest unified streaming STT API."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Smallest STT")

            # `model` is required by this endpoint -- a missing/invalid value
            # is rejected with a 400 before the WebSocket upgrades.
            query_params = {
                "model": self._settings.model,
                "language": self._settings.language,
                "encoding": self._encoding,
                "sample_rate": str(self.sample_rate),
                "word_timestamps": str(self._settings.word_timestamps).lower(),
                "full_transcript": str(self._settings.full_transcript).lower(),
                "sentence_timestamps": str(self._settings.sentence_timestamps).lower(),
                "redact_pii": str(self._settings.redact_pii).lower(),
                "redact_pci": str(self._settings.redact_pci).lower(),
                "diarize": str(self._settings.diarize).lower(),
            }

            ws_url = f"{self._base_url}/waves/v1/stt/live?{urlencode(query_params)}"

            self._websocket = await websocket_connect(
                ws_url,
                additional_headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "X-Source": "pipecat",
                    "X-Pipecat-Version": pipecat_version(),
                },
            )
            await self._call_event_handler("on_connected")
            logger.debug("Connected to Smallest STT")
        except Exception as e:
            await self.push_error(error_msg=f"Smallest STT connection error: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Flush and close the WebSocket connection."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug("Disconnecting from Smallest STT")
                await self._websocket.send(json.dumps({"type": "close_stream"}))
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Receive and process messages from the Smallest Pulse WebSocket."""
        async for message in self._get_websocket():
            try:
                data = json.loads(message)
                await self._process_response(data)
            except json.JSONDecodeError:
                logger.warning(f"{self} received non-JSON message: {message}")
            except Exception as e:
                logger.error(f"{self} error processing message: {e}")

    async def _process_response(self, data: dict):
        """Process a transcription or error response from the streaming API."""
        if data.get("type") == "error":
            await self.push_error(error_msg=f"Smallest STT error: {data.get('message', data)}")
            return

        is_final = data.get("is_final", False)
        text = (data.get("transcript") or data.get("transcription") or "").strip()

        if not text:
            return

        if is_final:
            await self.stop_processing_metrics()
            logger.debug(f"Smallest final transcript: [{text}]")
            await self._handle_transcription(text, True, data.get("language"))
            await self.push_frame(
                TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    data.get("language"),
                    result=data,
                )
            )
        else:
            logger.trace(f"Smallest interim transcript: [{text}]")
            await self.push_frame(
                InterimTranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    data.get("language"),
                    result=data,
                )
            )

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        """Handle a transcription result with tracing."""
        pass