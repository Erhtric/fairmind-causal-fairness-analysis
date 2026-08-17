"""Tests for llm.py module schema and summarization functions."""

from unittest.mock import MagicMock

from src.llm import (
    EvaluationQuestion,
    FairnessReport,
    LLMReportResult,
    generate_report_from_file_id,
    summarize_fairmind,
)


def test_evaluation_question_schema():
    eq = EvaluationQuestion(
        question="1. Which group has a higher probability/mean of positive outcome Y?",
        answer="(1) Group x1 has a higher probability",
    )
    assert eq.question == "1. Which group has a higher probability/mean of positive outcome Y?"
    assert eq.answer == "(1) Group x1 has a higher probability"


def test_fairness_report_schema():
    eq = EvaluationQuestion(
        question="2. What is the directional effect of intervening on X (x0 -> x1) on outcome Y?",
        answer="(1) Intervening on X increases Y",
    )
    report = FairnessReport(
        latex="\\documentclass{article}",
        evaluation=[eq],
    )
    assert report.latex == "\\documentclass{article}"
    assert len(report.evaluation) == 1
    assert report.evaluation[0].answer == "(1) Intervening on X increases Y"


def test_llm_report_result():
    eq = EvaluationQuestion(
        question="1. Which group has a higher probability/mean of positive outcome Y?",
        answer=True,
    )
    res = LLMReportResult(
        model="gpt-5.4-nano",
        effort="high",
        text="Text report",
        latex="Latex report",
        evaluation=[eq],
        usage={"total_tokens": 100},
    )
    text, latex, evaluation, usage = res
    assert text == "Text report"
    assert latex == "Latex report"
    assert len(evaluation) == 1
    assert evaluation[0].question == "1. Which group has a higher probability/mean of positive outcome Y?"
    assert usage == {"total_tokens": 100}


def test_summarize_fairmind():
    mock_client = MagicMock()
    eq = EvaluationQuestion(
        question="1. Which group has a higher probability/mean of positive outcome Y?",
        answer=True,
    )
    mock_parsed = MagicMock()
    mock_parsed.latex = "\\subsection*{1. Overview of the Fairness Analysis}\nSample Text"
    mock_parsed.evaluation = [eq]

    mock_resp = MagicMock()
    mock_resp.output_parsed = mock_parsed
    mock_resp.usage = {"tokens": 50}
    mock_client.responses.parse.return_value = mock_resp

    results_payload = {"dataset": "synthetic", "results": {}}
    res = summarize_fairmind(
        results=results_payload,
        client=mock_client,
        model="test-model",
        prompt_path="fairmind_v2.txt",
    )

    assert res.model == "test-model"
    assert "Sample Text" in res.text
    assert "\\documentclass" in res.latex
    assert res.evaluation == [eq]
    assert res.usage == {"tokens": 50}


def test_generate_report_from_file_id():
    mock_client = MagicMock()
    eq = EvaluationQuestion(
        question="1. Which group has a higher probability/mean of positive outcome Y?",
        answer=True,
    )
    mock_parsed = MagicMock()
    mock_parsed.latex = "\\subsection*{1. Overview of the Fairness Analysis}\nSample Text"
    mock_parsed.evaluation = [eq]

    mock_resp = MagicMock()
    mock_resp.output_parsed = mock_parsed
    mock_resp.usage = {"tokens": 50}
    mock_client.responses.parse.return_value = mock_resp

    prompt_kwargs = {
        "exposure": "X",
        "outcome": "Y",
        "x0": 0,
        "x1": 1,
        "yt": 1,
    }
    res = generate_report_from_file_id(
        file_id="file-123",
        client=mock_client,
        model="test-model",
        prompt_kwargs=prompt_kwargs,
    )

    assert res.model == "test-model"
    assert "Sample Text" in res.text
    assert "\\documentclass" in res.latex
    assert res.evaluation == [eq]
    assert res.usage == {"tokens": 50}
