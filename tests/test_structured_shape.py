"""The provider's top-level shape has to match the schema it was given.

A schema declaring OBJECT is sometimes answered with a one-element array wrapping
that object. Callers do data.get(...) and die with "'list' object has no attribute
'get'". That failed the citation verifier three attempts running and took a whole
pipeline to FAILED — from a well-formed answer in the wrong wrapper.
"""

import pytest

from livingbook.llm.gemini import _conform_to_schema_shape

OBJECT = {"type": "OBJECT"}
ARRAY = {"type": "ARRAY"}


def test_an_object_wrapped_in_an_array_is_unwrapped():
    assert _conform_to_schema_shape([{"verdict": "accept"}], OBJECT) == {
        "verdict": "accept"}


def test_an_object_that_is_already_an_object_is_untouched():
    payload = {"verdict": "reject", "identity_confirmed": False}
    assert _conform_to_schema_shape(payload, OBJECT) is payload


def test_several_objects_merge_with_the_first_winning():
    """The first answer is the primary one; later ones only fill gaps."""
    out = _conform_to_schema_shape(
        [{"verdict": "accept", "note": "first"}, {"verdict": "reject", "extra": "x"}],
        OBJECT)
    assert out["verdict"] == "accept"
    assert out["note"] == "first"
    assert out["extra"] == "x"


def test_a_bare_item_is_wrapped_when_an_array_was_asked_for():
    assert _conform_to_schema_shape({"items": [1, 2]}, ARRAY) == [1, 2]
    assert _conform_to_schema_shape({"a": 1}, ARRAY) == [{"a": 1}]


def test_a_correct_array_is_untouched():
    payload = [1, 2, 3]
    assert _conform_to_schema_shape(payload, ARRAY) is payload


@pytest.mark.parametrize("value", [[], {}])
def test_empty_containers_are_handled_without_crashing(value):
    # An empty list cannot become an object; an empty object already is one.
    result = _conform_to_schema_shape(value, OBJECT)
    assert result is None or result == {}


# -- a shape that cannot be reshaped is a schema violation --------------------
#
# The real failure was not an object in a wrapper. The model returned a bare list of
# caveat strings where the whole verification object was asked for. Nothing in that
# can be reshaped into the answer, so passing it through hands every consumer a list
# to call .get() on. It has to go back through the repair pass instead.

def test_a_list_with_no_object_cannot_be_conformed():
    assert _conform_to_schema_shape(
        ["The claim's mention of 'anomalies' should be understood as ..."],
        OBJECT) is None


def test_a_scalar_where_an_object_was_asked_for_cannot_be_conformed():
    assert _conform_to_schema_shape("not an object", OBJECT) is None
    assert _conform_to_schema_shape(7, OBJECT) is None


def test_none_stays_none():
    assert _conform_to_schema_shape(None, OBJECT) is None


def test_the_provider_repairs_rather_than_returning_a_bad_shape():
    """Conformance must run before the None check that triggers repair."""
    import inspect

    from livingbook.llm.gemini import GeminiProvider

    src = inspect.getsource(GeminiProvider.generate_structured)
    conform_at = src.find("_conform_to_schema_shape")
    repair_at = src.find("if parsed is None:")
    assert conform_at != -1 and repair_at != -1
    assert conform_at < repair_at, (
        "a wrong shape must be detected before the repair check, or it is returned "
        "to the caller unrepaired")
