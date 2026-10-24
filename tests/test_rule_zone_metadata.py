"""Optional zone metadata is additive and independent of plugin execution."""

import pytest
from fastapi.testclient import TestClient

from guard.api.app import create_app
from guard.pipeline.testing import make_settings
from guard.rules import base


LEGACY_TYPES = {
    "zone_intrusion", "ppe_missing", "fall_immobile", "vehicle_proximity", "running",
    "phone_usage", "smoking_fire", "fence_climb", "height_no_harness", "unsafe_carry",
}
SINGLE_ZONE_TYPES = {"zone_dwell", "wrong_way", "zone_overcapacity"}
SINGLE_ZONE_SCHEMA = {"type": "array", "minItems": 1, "maxItems": 1}


def test_legacy_metadata_keeps_exact_previous_fields_and_values():
    metadata = {item["type"]: item for item in base.available_rules()}
    assert len(LEGACY_TYPES) == 10
    assert LEGACY_TYPES <= metadata.keys()
    for rule_type in LEGACY_TYPES:
        cls = base.get_rule(rule_type)
        assert cls.zone_schema is None
        assert metadata[rule_type] == {
            "type": cls.type,
            "title": cls.title,
            "required_models": sorted(cls.required_models),
            "default_severity": cls.default_severity.value,
            "params_schema": cls.params_schema,
        }


def test_exactly_three_plugins_declare_one_explicit_zone():
    metadata = base.available_rules()
    declared = {item["type"]: item["zone_schema"] for item in metadata if "zone_schema" in item}
    assert declared == {name: SINGLE_ZONE_SCHEMA for name in SINGLE_ZONE_TYPES}
    for rule_type in SINGLE_ZONE_TYPES:
        assert base.get_rule(rule_type).zone_schema == SINGLE_ZONE_SCHEMA


@pytest.mark.parametrize("schema", [None, {"type": "array", "minItems": 2, "maxItems": 3}])
def test_metadata_uses_optional_class_contract_without_instantiating_plugin(monkeypatch, schema):
    base.load_plugins()

    class MetadataOnlyRule(base.RulePlugin):
        type = "unit_zone_metadata"
        title = "Metadata-only plugin"
        zone_schema = schema

        def __init__(self, *args, **kwargs):
            pytest.fail("Reading metadata must not instantiate a rule")

        def evaluate(self, ctx):
            pytest.fail("Reading metadata must not evaluate a frame")

    monkeypatch.setitem(base._REGISTRY, MetadataOnlyRule.type, MetadataOnlyRule)
    item = next(row for row in base.available_rules() if row["type"] == MetadataOnlyRule.type)
    if schema is None:
        assert "zone_schema" not in item
    else:
        assert item["zone_schema"] == schema


def test_http_rule_types_serializes_optional_zone_schema_without_runtime(tmp_path):
    # No lifespan: metadata route needs neither an engine nor source/model startup.
    app = create_app(make_settings(tmp_path, autostart=False), engine=object())
    client = TestClient(app)
    try:
        response = client.get("/api/rules/types")
    finally:
        client.close()
    assert response.status_code == 200
    assert response.json() == base.available_rules()
    metadata = {item["type"]: item for item in response.json()}
    assert all("zone_schema" not in metadata[name] for name in LEGACY_TYPES)
    assert all(metadata[name]["zone_schema"] == SINGLE_ZONE_SCHEMA for name in SINGLE_ZONE_TYPES)
