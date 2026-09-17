- Fixed STT TTFB reporting a wait other than the one it measures.

  - An `InterruptionFrame` no longer ends or discards an armed STT clock, so the
    utterance's own final still closes it.
  - A finalized transcript arriving before the VAD stop is reported at the stop
    and consumed there, rather than leaving a clock nothing can close.
  - A stop with no speech start before it takes over the earlier stop's reporter.
  - Deepgram's empty Finalize acknowledgement reports the transcript it
    acknowledges rather than the round trip to itself.
  - Speech end comes from the silence the analyzer counted, exposed as
    `VADAnalyzer.stop_debounce_secs` and carried on
    `VADUserStoppedSpeakingFrame.debounce_secs` (`None` when a producer does not
    report one), because `params.stop_secs` may be held below the real debounce.
