# Rule configuration validation before persistence

Camera creation/update now constructs each known registered plugin as
`get_rule(rule.type)(rule, camera)` during `PipelineEngine.validate_camera`.
This includes configured disabled rules and disabled cameras: a configuration
must not silently store parameters that its plugin cannot construct later.

The order remains camera/zone checks, unique rule IDs, known type, zone
references and rule window checks. Constructor validation follows those checks,
before the existing enabled-rule required-model check. Only after all validation
succeeds can camera CRUD persist YAML or stop/start workers.

`ValueError`, `TypeError` and `OverflowError` from the constructor become
`CameraError` containing the rule ID and type. Existing HTTP POST/PUT camera
routes return422 for that error. A failed PUT preserves the existing YAML bytes,
camera/runtime objects, active workers and event state. The camera API currently
has no PATCH route: PATCH returns405 and does not change configuration. This
feature does not add an endpoint.

Constructors must be pure configuration validation/initialization: numeric/type
checks, zone selection and fresh in-memory state are allowed; file/network I/O,
model loading, source/worker threads and frame `evaluate()` are not. The ten
existing constructors were inspected and are covered by a model-free regression
that blocks I/O, thread start, detector loading and frame evaluation. Temporary
validation instances are discarded; runtime workers still create their own
fresh plugin instances.

`params_schema` remains descriptive metadata used by the API/UI. There is no
new JSON-schema dependency or blanket schema interpreter. A strict plugin must
enforce its parameter contract in its constructor; legacy constructor behavior
is preserved. This check proves configuration can be constructed, not model
availability, supported classes, observable input or rule/model quality. It does
not change thresholds, EventManager semantics, model registries or defaults.
