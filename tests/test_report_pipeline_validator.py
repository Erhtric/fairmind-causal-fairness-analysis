"""Tests for the report scorer in src/report_pipeline/validator.py.

The scorer is the measuring instrument of the reporting experiment: the score
it produces is what gets reported, so a defect here does not cause a visible
failure, it causes a wrong number that looks right.  Two such defects have
already occurred (the one-argument spurious effect, and the direct/reverse
indirect effect), and neither produced any observable symptom.

The tests are organised as: threshold rules first (with boundary cases, since
every rule is a strict inequality), then the LaTeX answer parser, then scoring,
then a regression test pinning the published Adult figures.
"""

import numpy as np
import pytest

from src.report_pipeline.validator import (
    ANSWER_FALSE,
    ANSWER_TRUE,
    EPS_IE,
    GROUND_TRUTH_RULES,
    IE_DECOMPOSITION_KEY,
    N_QUESTIONS,
    THRESH_DE,
    THRESH_SE_REL,
    THRESH_TV,
    decomposition_ie,
    gt_q1_direct_discrimination,
    gt_q2_ie_mitigates_te,
    gt_q3_tv_practically_relevant,
    gt_q4_se_substantial,
    gt_q5_de_dominant_over_ie,
    parse_recap_answers,
    score_report,
)

# The exact FairMind output for Adult, as reported in the thesis.  Values are
# quoted to six decimals, which is the precision the report itself carries.
ADULT_EFFECTS = {
    "TV": 0.193714,
    "TE": 0.184736,
    "SE": 0.008977,
    "DE": 0.138404,
    "IE": 0.016869,
    "IE_reverse": -0.046333,
}

# Q2 confirmed False (amplifies, does not mitigate) with the supervisor
# (Antonucci, 2026-08-20): TE = DE + IE_prof with IE_prof := -IE_{x1,x0},
# and on Adult IE_prof is concordant in sign with DE, so |TE| > |DE|.
ADULT_EXPECTED_ANSWERS = [ANSWER_TRUE, ANSWER_FALSE, ANSWER_TRUE, ANSWER_FALSE, ANSWER_TRUE]


def answer_line(value: str, label: str = "Answer") -> str:
    """One Recap Question line as it appears in a generated report."""
    return rf"\item Some question? \textbf{{{label}:}} {value}"


def report_with(answers: list[str], label: str = "Answer") -> str:
    """A minimal LaTeX document carrying the given answers, in order."""
    body = "\n".join(answer_line(a, label) for a in answers)
    return f"\\begin{{enumerate}}\n{body}\n\\end{{enumerate}}\n"


# ---------------------------------------------------------------------------
# Which indirect effect the rules read
# ---------------------------------------------------------------------------


def test_decomposition_ie_reads_the_reverse_form():
    """The rules must use IE_{x1,x0}, the form that closes the decomposition."""
    assert decomposition_ie(ADULT_EFFECTS) == ADULT_EFFECTS[IE_DECOMPOSITION_KEY]
    assert decomposition_ie(ADULT_EFFECTS) != ADULT_EFFECTS["IE"]


def test_decomposition_ie_refuses_to_fall_back_to_the_direct_form():
    """Regression test for a bug that changed an answer with no symptom.

    When only "IE" is available the honest outcome is a loud failure.  Silently
    using the direct form flips Q2, and the score would still look like a valid
    score.
    """
    without_reverse = {k: v for k, v in ADULT_EFFECTS.items() if k != IE_DECOMPOSITION_KEY}
    with pytest.raises(KeyError):
        decomposition_ie(without_reverse)


def test_the_two_indirect_forms_would_disagree_on_q2():
    """Documents why the distinction matters, in the terms of the answer.

    This is the whole reason the key is explicit: on Adult the two forms give
    opposite answers to Q2. Confirmed with the supervisor (Antonucci,
    2026-08-20): the reverse form is the one that enters the additive
    decomposition TE = DE + (-IE_{x1,x0}), and on Adult it has sign opposite
    to DE, i.e. the mediator AMPLIFIES the total rather than mitigating it.
    """
    reverse = dict(ADULT_EFFECTS)
    direct = {**ADULT_EFFECTS, IE_DECOMPOSITION_KEY: ADULT_EFFECTS["IE"]}
    assert gt_q2_ie_mitigates_te(reverse) is False
    assert gt_q2_ie_mitigates_te(direct) is True


def test_adult_decomposition_identity_closes_on_the_reverse_form():
    """DE - IE_{x1,x0} = TE, to the precision of the published figures.

    The tolerance is 2e-6 rather than something tighter because the constants
    above are the six-decimal values as quoted, not full precision.
    """
    residual = ADULT_EFFECTS["DE"] - ADULT_EFFECTS["IE_reverse"] - ADULT_EFFECTS["TE"]
    assert abs(residual) < 2e-6


# ---------------------------------------------------------------------------
# Threshold rules, with boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "de, expected",
    [
        (THRESH_DE + 0.01, True),
        (-(THRESH_DE + 0.01), True),  # the rule is on the magnitude
        (THRESH_DE - 0.01, False),
        (THRESH_DE, False),  # strict inequality: exactly on the threshold is False
        (0.0, False),
    ],
)
def test_q1_direct_discrimination_threshold(de, expected):
    assert gt_q1_direct_discrimination({"DE": de}) is expected


@pytest.mark.parametrize(
    "ie, de, expected",
    [
        # Confirmed with the supervisor (Antonucci, 2026-08-20): with
        # IE_prof := -IE_{x1,x0}, TE = DE + IE_prof. DE and IE_prof
        # concordant in sign means the mediator AMPLIFIES the total; since
        # decomposition_ie() returns IE_{x1,x0} = -IE_prof, that translates
        # to decomposition_ie() and DE being concordant for MITIGATION, i.e.
        # the same sign as DE, not the opposite -- every case below was
        # checked by running gt_q2_ie_mitigates_te(), not derived on paper,
        # since an earlier draft of this same rule had the sign backwards.
        (0.05, 0.10, True),  # same sign as DE: mitigates
        (-0.05, -0.10, True),  # same sign, both negative: mitigates
        (0.05, -0.10, False),  # opposite sign: amplifies
        (-0.05, 0.10, False),  # opposite sign, the pattern seen on Adult: amplifies
        (EPS_IE / 2, 0.10, False),  # negligible channel, regardless of sign
        # The negligibility test is |IE| < EPS_IE, so a channel sitting exactly
        # on the floor is NOT discarded: the rule goes on to compare signs.
        (EPS_IE, 0.10, True),
        (0.0, 0.10, False),
    ],
)
def test_q2_mitigation_requires_the_same_sign_as_de_and_a_non_negligible_channel(
    ie, de, expected
):
    effects = {"DE": de, IE_DECOMPOSITION_KEY: ie}
    assert gt_q2_ie_mitigates_te(effects) is expected


@pytest.mark.parametrize(
    "tv, expected",
    [
        (THRESH_TV + 0.01, True),
        (-(THRESH_TV + 0.01), True),
        (THRESH_TV, False),
        (0.0, False),
    ],
)
def test_q3_total_variation_relevance_threshold(tv, expected):
    assert gt_q3_tv_practically_relevant({"TV": tv}) is expected


@pytest.mark.parametrize(
    "tv, se, expected",
    [
        (0.20, 0.05, True),  # ratio 0.25, above THRESH_SE_REL
        (0.20, 0.01, False),  # ratio 0.05, below
        # Exactly on the threshold, with values chosen so the ratio is exactly
        # representable: 0.20 * THRESH_SE_REL would give 0.10000000000000002
        # and land just above it, testing nothing about the boundary.
        (1.0, THRESH_SE_REL, False),
        (0.20, -0.05, True),  # the rule is on magnitudes
    ],
)
def test_q4_spurious_share_is_relative_to_total_variation(tv, se, expected):
    assert gt_q4_se_substantial({"TV": tv, "SE": se}) is expected


@pytest.mark.parametrize(
    "se, expected",
    [(0.01, True), (0.0, False)],
)
def test_q4_degenerate_branch_when_total_variation_vanishes(se, expected):
    """With TV = 0 the ratio is undefined; any non-zero SE counts as substantial."""
    assert gt_q4_se_substantial({"TV": 0.0, "SE": se}) is expected


@pytest.mark.parametrize(
    "de, ie, expected",
    [
        (0.14, -0.046, True),
        (0.04, -0.046, False),
        (0.046, 0.046, False),  # equal magnitudes: strict, so not dominant
        (-0.14, 0.046, True),  # magnitudes again, not signs
    ],
)
def test_q5_direct_dominates_indirect_by_magnitude(de, ie, expected):
    effects = {"DE": de, IE_DECOMPOSITION_KEY: ie}
    assert gt_q5_de_dominant_over_ie(effects) is expected


def test_rule_list_is_in_template_order_and_complete():
    """The list order is the document order; a mismatch misaligns every answer."""
    assert GROUND_TRUTH_RULES == [
        gt_q1_direct_discrimination,
        gt_q2_ie_mitigates_te,
        gt_q3_tv_practically_relevant,
        gt_q4_se_substantial,
        gt_q5_de_dominant_over_ie,
    ]
    assert N_QUESTIONS == 5


# ---------------------------------------------------------------------------
# Parsing answers out of the generated LaTeX
# ---------------------------------------------------------------------------


def test_parses_english_answers():
    answers = parse_recap_answers(report_with(["YES", "NO", "YES", "NO", "YES"]))
    assert answers == [ANSWER_TRUE, ANSWER_FALSE, ANSWER_TRUE, ANSWER_FALSE, ANSWER_TRUE]


def test_parses_legacy_italian_reports():
    """Reports produced before the switch to English stay scoreable.

    The module claims this in a comment; the claim is worth a test, because the
    Italian runs are cited in the thesis and must remain reproducible.
    """
    answers = parse_recap_answers(
        report_with(["SI", "NO", "SI", "NO", "SI"], label="Risposta")
    )
    assert answers == [ANSWER_TRUE, ANSWER_FALSE, ANSWER_TRUE, ANSWER_FALSE, ANSWER_TRUE]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("YES", ANSWER_TRUE),
        ("yes", ANSWER_TRUE),
        ("Y", ANSWER_TRUE),
        ("TRUE", ANSWER_TRUE),
        ("SI", ANSWER_TRUE),
        ("S\u00cc", ANSWER_TRUE),  # accented, normalised through NFKD
        ("VERO", ANSWER_TRUE),
        ("NO", ANSWER_FALSE),
        ("no.", ANSWER_FALSE),  # trailing punctuation is stripped
        ("FALSE", ANSWER_FALSE),
        ("FALSO", ANSWER_FALSE),
        ("MAYBE", None),
        ("42", None),
    ],
)
def test_answer_token_normalisation(raw, expected):
    assert parse_recap_answers(report_with([raw]))[0] == expected


def test_missing_answers_are_padded_with_none():
    """A model that drops questions must not shorten the answer list."""
    answers = parse_recap_answers(report_with(["YES", "NO"]))
    assert answers == [ANSWER_TRUE, ANSWER_FALSE, None, None, None]


def test_extra_answers_are_truncated():
    """A model that invents a sixth question must not lengthen it either."""
    answers = parse_recap_answers(report_with(["YES"] * 7))
    assert len(answers) == N_QUESTIONS


@pytest.mark.parametrize(
    "rendered, expected",
    [
        (r"\textbf{YES}", ANSWER_TRUE),
        (r"\emph{NO}", ANSWER_FALSE),
        (r"\textbf{\large YES}", ANSWER_TRUE),
        (r"{\bf NO}", ANSWER_FALSE),
    ],
)
def test_answer_wrapped_in_markup_is_still_read(rendered, expected):
    """A differently rendered answer must not be mistaken for a broken report.

    The pattern used to stop at the first backslash, so an answer wrapped in
    markup produced no match at all: the occurrence vanished, the list was
    padded with None, and five such answers scored 0/5.  In a thesis that
    number would read as a collapse of the model rather than a limitation of
    the parser, which is why this is now handled rather than documented.
    """
    latex = rf"\item Q? \textbf{{Answer:}} {rendered}"
    assert parse_recap_answers(latex)[0] == expected


def test_answer_followed_by_commentary_is_read_from_the_first_word():
    """Models append a justification despite being told to answer with a token."""
    latex = r"\item Q? \textbf{Answer:} YES (the direct effect is large)"
    assert parse_recap_answers(latex)[0] == ANSWER_TRUE


def test_commentary_before_the_answer_is_not_guessed_at():
    """When the answer does not come first, the safe outcome is unreadable.

    Picking a token out of the middle of a sentence would mean guessing; a
    None costs the question, which is the conservative direction.
    """
    latex = r"\item Q? \textbf{Answer:} The answer is YES"
    assert parse_recap_answers(latex)[0] is None


def test_trailing_latex_line_break_does_not_hide_the_answer():
    latex = r"\item Q? \textbf{Answer:} NO \\"
    assert parse_recap_answers(latex)[0] == ANSWER_FALSE


def test_answers_on_separate_lines_are_not_merged():
    """The capture stops at the newline, so one line cannot absorb the next."""
    latex = "\\textbf{Answer:} YES\n\\textbf{Answer:} NO\n"
    assert parse_recap_answers(latex)[:2] == [ANSWER_TRUE, ANSWER_FALSE]


def test_empty_document_yields_all_none():
    assert parse_recap_answers("") == [None] * N_QUESTIONS


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_perfect_report_scores_full_marks():
    report = score_report(report_with(ADULT_EXPECTED_ANSWERS), ADULT_EFFECTS)
    assert report.n_correct == N_QUESTIONS
    assert report.score == 1.0
    assert report.n_unparseable == 0


def test_every_answer_wrong_scores_zero():
    flipped = [ANSWER_FALSE if a == ANSWER_TRUE else ANSWER_TRUE for a in ADULT_EXPECTED_ANSWERS]
    report = score_report(report_with(flipped), ADULT_EFFECTS)
    assert report.n_correct == 0
    assert report.score == 0.0


def test_unreadable_answers_count_as_errors_never_as_credit():
    """A report that cannot be read must not score well for that reason."""
    answers = list(ADULT_EXPECTED_ANSWERS)
    answers[0] = "PERHAPS"
    report = score_report(report_with(answers), ADULT_EFFECTS)
    assert report.n_unparseable == 1
    assert report.n_correct == N_QUESTIONS - 1
    assert report.results[0].correct is False
    assert report.results[0].llm_answer is None


def test_score_report_indexes_questions_from_one():
    report = score_report(report_with(ADULT_EXPECTED_ANSWERS), ADULT_EFFECTS)
    assert [r.index for r in report.results] == [1, 2, 3, 4, 5]


def test_to_dict_is_serialisable_and_reports_the_expected_shape():
    """The dict is what lands in the result JSON kept as experimental evidence."""
    import json

    payload = score_report(report_with(ADULT_EXPECTED_ANSWERS), ADULT_EFFECTS).to_dict()
    assert payload["n_correct"] == N_QUESTIONS
    assert payload["score_pct"] == "100.0%"
    assert len(payload["questions"]) == N_QUESTIONS
    json.dumps(payload)  # must not raise


# ---------------------------------------------------------------------------
# Regression: the published Adult figures
# ---------------------------------------------------------------------------


def test_adult_ground_truth_vector_is_pinned():
    """Pins the expected answers behind every score quoted in the thesis.

    If a threshold is retuned or the indirect form is swapped, this test names
    the question that moved, instead of the change surfacing as a score that
    silently differs from the one already written up.
    """
    computed = [
        ANSWER_TRUE if rule(ADULT_EFFECTS) else ANSWER_FALSE for rule in GROUND_TRUTH_RULES
    ]
    assert computed == ADULT_EXPECTED_ANSWERS


def test_rules_work_on_the_numpy_types_the_pipeline_actually_produces():
    """FairMind returns mixed types, and the tests above only use plain floats.

    ``total_variation`` returns ``numpy.float64`` while ``natural_direct_effect``
    returns a Python ``float``, so the dict reaching the scorer is mixed.  The
    consequence is narrow but real: on a numpy input a rule returns
    ``numpy.bool_``, and ``numpy.bool_(True) is True`` is False.  Scoring is
    safe because it goes through truthiness, and this test pins that, so an
    identity comparison introduced later fails here rather than in a run.
    """
    numpy_effects = {k: np.float64(v) for k, v in ADULT_EFFECTS.items()}

    for rule in GROUND_TRUTH_RULES:
        assert bool(rule(numpy_effects)) in (True, False)

    computed = [
        ANSWER_TRUE if rule(numpy_effects) else ANSWER_FALSE for rule in GROUND_TRUTH_RULES
    ]
    assert computed == ADULT_EXPECTED_ANSWERS


def test_scoring_is_unaffected_by_the_numeric_type_of_the_effects():
    """The same report must score identically on float and numpy.float64."""
    document = report_with(ADULT_EXPECTED_ANSWERS)
    numpy_effects = {k: np.float64(v) for k, v in ADULT_EFFECTS.items()}
    assert (
        score_report(document, numpy_effects).to_dict()
        == score_report(document, ADULT_EFFECTS).to_dict()
    )


def test_adult_spurious_share_sits_below_the_threshold_that_decides_q4():
    """Q4 is the only NO, and it is not a close call in the direction of noise.

    Recording the actual ratio makes the margin visible: if a future run drifts
    towards the threshold, the number in this assertion is where to look.
    """
    ratio = abs(ADULT_EFFECTS["SE"]) / abs(ADULT_EFFECTS["TV"])
    assert ratio == pytest.approx(0.046, abs=0.001)
    assert ratio < THRESH_SE_REL
