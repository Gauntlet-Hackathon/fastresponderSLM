"""labels.py must mirror the vendored `extracted` schema exactly."""

import hashlib
import json

from rescue_tagger import labels as L
from rescue_tagger.records import SCHEMA_DIR

SCHEMA = json.loads((SCHEMA_DIR / "payloads" / "extracted.schema.json").read_text("utf-8"))
PROPS = SCHEMA["properties"]


def _enum(*path: str) -> tuple[str, ...]:
    node = PROPS
    for key in path:
        node = node[key]
    return tuple(node["enum"])


def test_enums_match_schema():
    assert _enum("vulnerable", "items", "properties", "kind") == L.VULNERABLE_KINDS
    assert _enum("hazards", "items", "properties", "kind") == L.HAZARD_KINDS
    assert _enum("needs", "items", "properties", "kind") == L.NEED_KINDS
    assert _enum("request_kind") == L.REQUEST_KINDS
    assert _enum("location", "properties", "type") == L.LOCATION_TYPES


def test_bio_tags_cover_every_span_label():
    assert L.BIO_TAGS[0] == "O"
    assert len(L.BIO_TAGS) == 1 + 2 * len(L.SPAN_LABELS)
    assert all(L.ID2TAG[L.TAG2ID[t]] == t for t in L.BIO_TAGS)


def test_vendored_schemas_unchanged():
    """Bump these hashes only together with a deliberate re-copy (see schema/README.md)."""
    expected = {
        "common.schema.json": "29ab356b2e296381e87f484ea71bd1fe39f6028a8890260bed5dc2e2c1d82838",
        "payloads/extracted.schema.json":
            "e33fbc7a05c554891eec3495a266724b3144a4598a622d9a050c939376ab259f",
    }
    for rel, digest in expected.items():
        assert hashlib.sha256((SCHEMA_DIR / rel).read_bytes()).hexdigest() == digest, rel
