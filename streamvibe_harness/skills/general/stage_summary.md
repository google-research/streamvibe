# skill: stage_summary status: active family: procedural_summarization
trigger: summarize progress | recap completed and remaining stages | provide progress updates
watch: evidence-grounded completed stages, current stage, unresolved deviation, and explicitly supported remaining work
speak: a stage completes | the user asks for a recap | the task pauses at a meaningful boundary
silent: the task is mid-stage with no new progress | the same stage state was already summarized
cadence: stage-boundary driven | summarize again only after progress changes or a new request
answer: state what is visibly complete and what remains supported by the request or procedure state
