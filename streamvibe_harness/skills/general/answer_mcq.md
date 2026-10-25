# skill: answer_mcq status: active family: evidence_selection
trigger: multiple-choice question | options A B C D | respond with the selected option
watch: the visible evidence required to distinguish the options | whether enough evidence has accumulated
speak: the evidence supports one option | the stream ends with sufficient evidence for a decision
silent: the evidence is insufficient | the question was already answered and no relevant evidence changed
cadence: wait for sufficient evidence | answer once unless the question or supporting evidence changes
answer: return only the option format requested by the user
