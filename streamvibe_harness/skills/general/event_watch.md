# skill: event_watch status: active family: event_detection
trigger: notify when a target appears starts stops enters leaves or changes state
watch: the exact requested target and transition | its first confirmed occurrence after the request became active
speak: the requested transition becomes clearly visible and has not been reported for this occurrence
silent: the target is absent or ambiguous | a similar but different event occurs | the same occurrence was already reported
cadence: event-edge driven | report each distinct confirmed occurrence once
answer: report the detected transition concisely in the user's language
