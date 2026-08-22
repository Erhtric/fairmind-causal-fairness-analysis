"""Tests for the LLM boundary and the post-hoc annotation step.

Two concerns live here.

The first is cleaning up whatever the model returns: markdown fences, prose
wrapped around the document, placeholders left unfilled.  These are pure text
functions and need no network.

The second is the request itself.  ``call_llm_report`` is not tested against a
real endpoint; a stub records the arguments instead.  What matters is not that
the call succeeds but that it carries ``temperature=0`` and
``cache_prompt=False``: the byte-identical reproducibility claimed for these
runs rests entirely on those two parameters, and nothing else in the codebase
would notice if they changed.
"""

import json

import pytest

from src.report_pipeline import llm_client
from src.report_pipeline.annotate import annotate_recap_answers
from src.report_pipeline.llm_client import (
    call_llm_report,
    extract_latex_document,
    find_unfilled_placeholders,
    strip_markdown_fences,
)
from src.report_pipeline.prompt_builder import build_prompts
from src.report_pipeline.validator import ANSWER_FALSE, ANSWER_TRUE, score_report

MINIMAL_DOC = "\\documentclass{article}\n\\begin{document}\nHello.\n\\end{document}"


# ---------------------------------------------------------------------------
# Cleaning up model output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fence", ["```latex", "```tex", "```"])
def test_strip_markdown_fences_removes_every_fence_flavour(fence):
    wrapped = f"{fence}\n{MINIMAL_DOC}\n```"
    assert strip_markdown_fences(wrapped) == MINIMAL_DOC


def test_strip_markdown_fences_leaves_unfenced_text_alone():
    assert strip_markdown_fences(MINIMAL_DOC) == MINIMAL_DOC


def test_strip_markdown_fences_leaves_an_unclosed_fence_alone():
    """Half a fence is not a fence; better to keep the text than to mangle it."""
    text = f"```latex\n{MINIMAL_DOC}"
    assert strip_markdown_fences(text).startswith("```latex")


def test_extract_latex_document_discards_surrounding_prose():
    """Models add a preamble and a sign-off despite being told not to."""
    noisy = f"Certainly! Here is the report:\n\n{MINIMAL_DOC}\n\nLet me know if..."
    assert extract_latex_document(noisy) == MINIMAL_DOC


def test_extract_latex_document_handles_prose_inside_a_fence():
    noisy = f"```latex\nBlah\n{MINIMAL_DOC}\n```"
    assert extract_latex_document(noisy) == MINIMAL_DOC


def test_extract_latex_document_returns_the_text_when_there_is_no_document():
    """Without a document to isolate, the caller gets the cleaned text back.

    The malformed output then fails downstream, at the placeholder check or the
    scorer, which is where the failure is legible.
    """
    assert extract_latex_document("I cannot help with that.") == "I cannot help with that."


def test_extract_latex_document_keeps_the_last_end_document():
    """A model that repeats \\end{document} must not truncate the real one."""
    doubled = MINIMAL_DOC + "\ntrailing\n\\end{document}"
    assert extract_latex_document(doubled).endswith("\\end{document}")
    assert "trailing" in extract_latex_document(doubled)


# ---------------------------------------------------------------------------
# Unfilled placeholders
# ---------------------------------------------------------------------------


def test_finds_well_formed_placeholders():
    assert find_unfilled_placeholders("a <<ANSWER_Q1>> b <<QUALITATIVE_DE>>") == [
        "ANSWER_Q1",
        "QUALITATIVE_DE",
    ]


def test_finds_a_malformed_placeholder_too():
    """Regression test: the escaped-underscore token that shipped in a report.

    The previous pattern was ``<<([A-Z0-9_]+)>>``, which does not match a
    backslash, so ``<<PROTECTED\\_ATTR>>`` was reported as "nothing left to
    fill" while sitting unfilled in the document, where the model then invented
    a value for it.  A token nobody filled is an unfilled token, whatever its
    spelling.
    """
    assert find_unfilled_placeholders(r"x <<PROTECTED\_ATTR>> y") == [r"PROTECTED\_ATTR"]


def test_the_literal_example_in_the_prompt_is_not_a_placeholder():
    """Regression test for a false positive of the permissive pattern.

    The instructions sent to the model contain the literal string ``<<...>>``
    as an example of what a placeholder looks like. A pattern wide enough to
    match anything between the angle brackets reported it as an unfilled
    placeholder, which made the notebook print a ninth slot that does not
    exist. Dots and spaces are therefore excluded from the name.
    """
    prompt_text = "Fill in the eight remaining <<...>> placeholders as instructed."
    assert find_unfilled_placeholders(prompt_text) == []


def test_a_complete_document_reports_nothing_left():
    assert find_unfilled_placeholders(MINIMAL_DOC) == []


def test_placeholder_search_does_not_span_lines():
    """``<<`` and ``>>`` on different lines are not a placeholder.

    Without the newline exclusion the pattern would swallow whole paragraphs
    between an unrelated ``<<`` and a later ``>>``.
    """
    assert find_unfilled_placeholders("a << b\nc >> d") == []


# ---------------------------------------------------------------------------
# The request contract
# ---------------------------------------------------------------------------


class _RecordingClient:
    """Stands in for openai.OpenAI and records how it was called."""

    def __init__(self, **init_kwargs):
        self.init_kwargs = init_kwargs
        _RecordingClient.last = self
        self.chat = self  # chat.completions.create resolves back to this object
        self.completions = self

    def create(self, **kwargs):
        self.call_kwargs = kwargs
        return _Response()


class _Usage:
    prompt_tokens, completion_tokens, total_tokens = 1253, 738, 1991


class _Message:
    content = MINIMAL_DOC


class _Choice:
    finish_reason = "stop"
    message = _Message()


class _Response:
    usage = _Usage()
    choices = [_Choice()]


@pytest.fixture
def recorded(monkeypatch):
    """Run call_llm_report against the stub and return the recorded request."""
    monkeypatch.setattr(llm_client, "OpenAI", _RecordingClient)
    config = {
        "model": "qwen2.5-14b-instruct",
        "base_url": "http://testhost:8080/v1",
        "api_key": "not-needed",
    }
    latex, usage, elapsed = call_llm_report("SYS", "USER", config=config)
    return _RecordingClient.last, latex, usage, elapsed


def test_request_is_deterministic_by_construction(recorded):
    """The reproducibility claim reduces to these two parameters."""
    client, _, _, _ = recorded
    assert client.call_kwargs["temperature"] == 0
    assert client.call_kwargs["extra_body"] == {"cache_prompt": False}


def test_cache_prompt_can_be_re_enabled_but_is_off_by_default(monkeypatch):
    """The default is what protects the runs; the override stays available."""
    monkeypatch.setattr(llm_client, "OpenAI", _RecordingClient)
    config = {"model": "m", "base_url": "u", "api_key": "k"}
    call_llm_report("SYS", "USER", config=config, cache_prompt=True)
    assert _RecordingClient.last.call_kwargs["extra_body"] == {"cache_prompt": True}


def test_both_prompts_are_sent_in_the_expected_roles(recorded):
    client, _, _, _ = recorded
    assert client.call_kwargs["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]


def test_a_truncated_response_raises_instead_of_being_scored(monkeypatch):
    """Truncation is a known failure mode of this model and must be loud.

    Without the guard the partial document flows on to the scorer and produces
    a number that looks like a result.  The message names both the reason and
    the token budget, since raising max_tokens is the usual remedy.
    """
    monkeypatch.setattr(llm_client, "OpenAI", _RecordingClient)
    monkeypatch.setattr(_Choice, "finish_reason", "length")
    config = {"model": "m", "base_url": "u", "api_key": "k"}

    with pytest.raises(ValueError, match="did not finish"):
        call_llm_report("SYS", "USER", config=config)


def test_truncation_can_be_inspected_on_purpose(monkeypatch):
    """The strict default protects runs; the escape hatch stays available."""
    monkeypatch.setattr(llm_client, "OpenAI", _RecordingClient)
    monkeypatch.setattr(_Choice, "finish_reason", "length")
    config = {"model": "m", "base_url": "u", "api_key": "k"}

    latex, usage, _ = call_llm_report(
        "SYS", "USER", config=config, require_complete=False
    )
    assert usage["finish_reason"] == "length"
    assert latex == MINIMAL_DOC


def test_usage_and_timing_are_reported(recorded):
    _, latex, usage, elapsed = recorded
    assert usage == {
        "input_tokens": 1253,
        "output_tokens": 738,
        "total_tokens": 1991,
        "finish_reason": "stop",
    }
    assert elapsed >= 0
    assert latex == MINIMAL_DOC


# ---------------------------------------------------------------------------
# Post-hoc annotation
# ---------------------------------------------------------------------------

ANNOTATION_EFFECTS = {
    "TV": 0.193714,
    "TE": 0.184736,
    "SE": 0.008977,
    "DE": 0.138404,
    "IE": 0.016869,
    "IE_reverse": -0.046333,
}


def _document(answers: list[str]) -> str:
    lines = "\n".join(
        rf"\item Question {i}? \textbf{{Answer:}} {a}" for i, a in enumerate(answers, 1)
    )
    return f"\\begin{{enumerate}}\n{lines}\n\\end{{enumerate}}\n\\end{{document}}"


def test_annotation_shows_both_answers_and_the_verdict():
    doc = _document([ANSWER_TRUE] * 5)
    score = score_report(doc, ANNOTATION_EFFECTS)
    annotated = annotate_recap_answers(doc, score)

    assert r"\textbf{LLM:}" in annotated
    assert r"\textbf{Expected:}" in annotated
    assert "(correct)" in annotated
    assert "(wrong)" in annotated  # Q4 is NO on Adult, so answering YES is wrong


def test_annotation_appends_the_score_before_end_document():
    doc = _document([ANSWER_TRUE, ANSWER_FALSE, ANSWER_TRUE, ANSWER_FALSE, ANSWER_TRUE])
    score = score_report(doc, ANNOTATION_EFFECTS)
    annotated = annotate_recap_answers(doc, score)

    assert "Answer consistency:} 5/5" in annotated
    assert annotated.index("Answer consistency") < annotated.index(r"\end{document}")
    assert annotated.rstrip().endswith(r"\end{document}")


def test_annotation_marks_an_unreadable_answer_as_such():
    doc = _document(["PERHAPS"] + [ANSWER_TRUE] * 4)
    score = score_report(doc, ANNOTATION_EFFECTS)
    annotated = annotate_recap_answers(doc, score)
    assert "(unreadable)" in annotated


def test_annotation_leaves_surplus_answers_untouched():
    """A malformed document must not be silently rewritten to look tidy."""
    doc = _document([ANSWER_TRUE] * 7)
    score = score_report(doc, ANNOTATION_EFFECTS)
    annotated = annotate_recap_answers(doc, score)
    assert annotated.count(r"\textbf{LLM:}") == 5
    assert annotated.count(r"\textbf{Answer:}") == 2


def test_annotation_without_end_document_still_appends_the_summary():
    doc = r"\item Q? \textbf{Answer:} YES"
    score = score_report(doc, ANNOTATION_EFFECTS)
    assert "Answer consistency" in annotate_recap_answers(doc, score)


# ---------------------------------------------------------------------------
# The methodological guarantee the score depends on
# ---------------------------------------------------------------------------


def test_the_model_never_receives_the_expected_answers():
    """The prompt must carry no answer, only the slots to fill.

    If an expected answer appeared in the template, the model could copy it and
    the score would stop measuring anything.  This is why annotation runs after
    generation and scoring rather than being baked into the template.
    """
    context = {
        "dataset": "adult",
        "protected_attr": "S2_gender",
        "x0": "Female",
        "x1": "Male",
        "outcome_attr": "T_income (>50K)",
        "mediator": "hours-per-week",
        "confounder": "education",
    }
    system, user = build_prompts(ANNOTATION_EFFECTS, context, "2026-08-16")

    # Every answer slot is still an unfilled placeholder ...
    for i in range(1, 6):
        assert f"<<ANSWER_Q{i}>>" in user
    # ... and no answer line has been pre-filled with a value.
    assert r"\textbf{Answer:} YES" not in user
    assert r"\textbf{Answer:} NO" not in user
    assert "Expected:" not in user and "Expected:" not in system


def test_annotator_and_validator_read_the_same_occurrences():
    """The two modules pair answers by position, so they must agree on the set.

    Regression test for a coupling defect: annotate.py held its own copy of the
    pattern.  When the validator was taught to read answers wrapped in markup
    and annotate.py was not, a document mixing the two renderings made the
    annotator skip the wrapped line and shift every later pairing, so Q4 was
    shown the expected answer belonging to Q3.
    """
    mixed = (
        "\\item Q1? \\textbf{Answer:} \\textbf{YES}\n"
        "\\item Q2? \\textbf{Answer:} NO\n"
        "\\item Q3? \\textbf{Answer:} YES\n"
        "\\item Q4? \\textbf{Answer:} NO\n"
        "\\item Q5? \\textbf{Answer:} YES\n"
        "\\end{document}"
    )
    score = score_report(mixed, ANNOTATION_EFFECTS)
    annotated = annotate_recap_answers(mixed, score)

    assert score.n_correct == 5
    # every answer line is rewritten, including the one wrapped in markup
    assert annotated.count(r"\textbf{LLM:}") == 5
    assert r"\textbf{Answer:}" not in annotated
    # and the pairing is not shifted: Q4 is the only NO on Adult
    q4_line = next(ln for ln in annotated.splitlines() if ln.startswith(r"\item Q4?"))
    assert r"\textbf{Expected:} NO" in q4_line


def test_annotation_result_is_consistent_with_the_score_json():
    """The annotated document and the stored JSON must tell the same story."""
    doc = _document([ANSWER_TRUE, ANSWER_TRUE, ANSWER_TRUE, ANSWER_FALSE, ANSWER_TRUE])
    score = score_report(doc, ANNOTATION_EFFECTS)
    payload = json.loads(json.dumps(score.to_dict()))
    annotated = annotate_recap_answers(doc, score)
    assert f"{payload['n_correct']}/{payload['n_total']}" in annotated
