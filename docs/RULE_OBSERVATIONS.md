# Explicit rule observation coverage

`FrameContext.model_observations` is a fresh dictionary for the current frame,
with values `observed`, `not_observable`, `unsupported`, or `error` per logical
model ID. Both it and the optional `rule_observations` dictionary default to
independent empty dictionaries, preserving existing context constructors.

`RulePlugin.observation_status(ctx) -> str` combines only `required_models`,
with priority **error > unsupported > not_observable > observed**. An absent
required-model status means `not_observable`; an invalid status value means
`error`. Empty `required_models` yields `observed`. The helper is pure: it does
not gate `evaluate()`, reset plugin state, publish events, or fill rule status.
New plugins opt in and can report their own result explicitly:

```python
status = self.observation_status(ctx)
ctx.rule_observations[self.cfg.rule_id] = status
if status != "observed":
    # Reset this plugin's temporal evidence as required by its own contract.
    return []
```

The plugin can refine coverage for its own geometry/history requirements.
Existing plugins retain their existing evaluation behavior and need not call
the helper. `unsupported` is an explicit capability result; neither an empty
model output nor a missing status implies that a capability is unsupported.

Unknown or failed observation is not proof of physical safety. Empty successful
inference is an observed model run, not proof of absent PPE or a safe person.
This feature does not change thresholds, model classes, rule min-duration,
EventManager `min_ratio`, close/cooldown policy, or historical metrics/GT.

## Runtime population

| Current-frame result | Model status |
|---|---|
| Normal successful inference, including zero boxes | `observed` |
| Inference exception, including asynchronous backend failure | `error` |
| Successful full-frame PPE inference, including zero boxes | `observed` |
| Person-crop PPE: at least one eligible observed track, all calls succeed | `observed` |
| Person-crop PPE: valid zero eligible tracks, including retained stale tracks | `not_observable` |
| Person-crop PPE: any failed/invalid ROI or incomplete call coverage | `error` |
| Model never called for this frame | No entry; helper treats missing as `not_observable` |

A generic inference exception is not reclassified as `unsupported`. Capability
producers can report that status explicitly. Rejected/ambiguous PPE boxes can
leave person attributes unknown despite a successful model observation; the
model status is not a substitute for per-person attribute coverage. Existing
crop masks still make failed/stale/invalid affected attributes `None` after
voting. An empty successful ROI contributes an unknown observation, never an
invented negative vote; prior temporal votes retain their original semantics.

`CameraWorker.last_context` holds the last completed frame context. The worker's
`status()` includes copies of its model/rule maps when that context exists;
there are no coverage entries before the first completed context. Every frame
gets new maps, and a durable capture reset clears `last_context` before the new
frame is inferred. Model failures remain in error metrics. A thrown plugin
evaluation is reported as rule `error`; plugins otherwise report explicitly.

Future S1/S2 evaluations propagate successful actual calls into their contexts
and include the maps in sample traces. These runners retain fail-fast inference
behavior: a failed call aborts the evaluation and does not become a successful
empty sample. S2 crop frames with no eligible tracks report `not_observable`.
No historical run is repeated or rewritten by this metadata addition.

Capture identity is not added to `FrameContext`: the worker's durable capture
boundary already closes old events before replacing trackers and all rule
instances. A new capture must not inherit temporal plugin state.
