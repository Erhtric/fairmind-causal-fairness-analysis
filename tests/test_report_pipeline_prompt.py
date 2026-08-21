"""Contract tests between template.tex and the code that fills it.

The template and prompt_builder.py are coupled by a set of literal ``<<NAME>>``
tokens that exist in two places at once: as text in the template, and as string
keys in the module.  Nothing in the language enforces that the two agree, and a
mismatch is silent: ``str.replace()`` on a token that is not there is a no-op,
so the value is computed, discarded, and the placeholder ships in the report.

That is exactly what happened when the template was translated to English and
the underscore inside two tokens was LaTeX-escaped (``<<PROTECTED\\_ATTR>>``).
The tests below pin the contract in both directions.
"""

import re

import pytest

from src.report_pipeline.prompt_builder import (
    _EFFECT_KEYS,
    _METADATA_KEYS,
    ANSWER_PLACEHOLDERS,
    QUALITATIVE_PLACEHOLDERS,
    build_prompts,
    escape_latex,
    load_template,
    prefill_template,
)
from src.report_pipeline.validator import GROUND_TRUTH_RULES, N_QUESTIONS

# Deliberately permissive: it matches ANY <<...>> token, including malformed
# ones such as <<PROTECTED\_ATTR>>.  A strict pattern here would reproduce the
# very blind spot these tests exist to catch.
ANY_PLACEHOLDER = re.compile(r"<<[^>]+>>")


@pytest.fixture
def effects() -> dict[str, float]:
    """The five effects, with the real Adult magnitudes."""
    return {
        "TV": 0.193714,
        "TE": 0.184736,
        "SE": 0.008977,
        "DE": 0.138404,
        "IE": -0.046333,
    }


@pytest.fixture
def context() -> dict[str, str]:
    """A configuration whose values contain LaTeX-hostile characters."""
    return {
        "dataset": "adult",
        "protected_attr": "S2_gender",
        "x0": "Female",
        "x1": "Male",
        "outcome_attr": "T_income (>50K)",
        "mediator": "hours-per-week",
        "confounder": "education",
    }


def test_every_placeholder_the_code_fills_exists_in_the_template():
    """Each token prompt_builder replaces must be present in template.tex.

    This is the direction that failed: the module kept replacing
    ``<<PROTECTED_ATTR>>`` long after the template had renamed it.
    """
    template = load_template()
    expected = _METADATA_KEYS + _EFFECT_KEYS
    missing = [key for key in expected if f"<<{key}>>" not in template]
    assert missing == [], (
        f"prompt_builder fills {missing}, but template.tex has no such token. "
        "The replace() is a silent no-op and the value is discarded."
    )


def test_every_placeholder_in_the_template_is_known_to_the_code():
    """The reverse direction: no orphan token may sit in the template.

    An unknown token is never filled by anyone, so it reaches the model, which
    is free to invent a value for it.
    """
    template = load_template()
    found = {m.strip("<>") for m in ANY_PLACEHOLDER.findall(template)}
    known = set(
        _METADATA_KEYS + _EFFECT_KEYS + QUALITATIVE_PLACEHOLDERS + ANSWER_PLACEHOLDERS
    )
    assert found - known == set(), (
        f"template.tex contains tokens nobody fills: {sorted(found - known)}"
    )


def test_prefill_leaves_exactly_the_placeholders_meant_for_the_model(effects, context):
    """After prefilling, only the 8 qualitative/answer tokens may survive.

    The system prompt tells the model the document "contains exactly eight
    placeholders".  If a ninth survives, that statement is false and the model
    has been handed a slot it was never told about.
    """
    prefilled = prefill_template(load_template(), effects, context, "2026-08-16")
    leftover = sorted(m.strip("<>") for m in ANY_PLACEHOLDER.findall(prefilled))
    expected = sorted(QUALITATIVE_PLACEHOLDERS + ANSWER_PLACEHOLDERS)
    assert leftover == expected


def test_prefilled_document_states_the_configured_names(effects, context):
    """The values must reach the document, not merely be computed.

    Asserting on the presence of the token is not enough: what matters is that
    the report names the actual protected attribute and outcome, since those
    identify what was analysed.
    """
    prefilled = prefill_template(load_template(), effects, context, "2026-08-16")
    assert r"S2\_gender" in prefilled
    assert r"T\_income (>50K)" in prefilled


def test_question_count_agrees_across_template_and_validator():
    """Three independent definitions of "how many questions" must coincide."""
    template = load_template()
    in_template = len(re.findall(r"<<ANSWER_Q\d+>>", template))
    assert in_template == N_QUESTIONS == len(GROUND_TRUTH_RULES) == len(
        ANSWER_PLACEHOLDERS
    )


# ---------------------------------------------------------------------------
# LaTeX escaping of injected values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, escaped",
    [
        ("S2_gender", r"S2\_gender"),  # the case that actually occurs
        ("a&b", r"a\&b"),
        ("100%", r"100\%"),
        ("x$y", r"x\$y"),
        ("#1", r"\#1"),
        ("a~b", r"a\textasciitilde{}b"),
        ("x^2", r"x\textasciicircum{}2"),
        ("\\", r"\textbackslash{}"),
        ("a_b_c", r"a\_b\_c"),  # every occurrence, not just the first
        ("plain text", "plain text"),
        ("", ""),
    ],
)
def test_escape_latex_handles_every_special_character(raw, escaped):
    assert escape_latex(raw) == escaped


def test_escape_latex_does_not_double_escape_its_own_output():
    """The backslashes it emits must not be re-escaped in the same pass.

    The function maps character by character and joins, rather than applying
    str.replace() in sequence.  A sequential implementation would rewrite the
    backslash of an already-escaped ``\\_`` and produce
    ``\\textbackslash{}_``.  This test locks the current implementation in.
    """
    assert escape_latex("a_b") == r"a\_b"
    assert r"\textbackslash" not in escape_latex("a_b&c%d")


def test_escape_latex_is_not_idempotent_and_must_be_applied_once():
    """Characterises a real hazard: escaping twice corrupts the value.

    Applying it again turns ``a\\_b`` into ``a\\textbackslash{}\\_b``, which
    renders as literal backslash text in the PDF.  prefill_template applies it
    exactly once, and nothing downstream may apply it again.
    """
    once = escape_latex("a_b")
    assert escape_latex(once) == r"a\textbackslash{}\_b"
    assert escape_latex(once) != once


def test_numeric_effects_are_formatted_to_six_decimals(effects, context):
    """The report quotes six decimals; the tests and the thesis assume it."""
    prefilled = prefill_template(load_template(), effects, context, "2026-08-16")
    assert "0.193714" in prefilled
    assert "-0.046333" in prefilled


def test_optional_context_keys_fall_back_instead_of_raising(effects):
    """A configuration with no mediator or confounder must still build."""
    minimal = {
        "dataset": "toy",
        "protected_attr": "X",
        "x0": "a",
        "x1": "b",
        "outcome_attr": "Y",
    }
    prefilled = prefill_template(load_template(), effects, minimal, "2026-08-16")
    assert "--" in prefilled


def test_build_prompts_returns_a_system_and_user_prompt(effects, context):
    system, user = build_prompts(effects, context, "2026-08-16")
    assert "causal fairness reporting assistant" in system
    assert r"\documentclass" in user
    assert "<<QUALITATIVE_TOTAL>>" in user
