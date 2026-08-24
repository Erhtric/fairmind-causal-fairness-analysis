"""LLM call for generating the LaTeX report.

``src.llm.call_llm()`` cannot be reused here: it parses a JSON block out of
the answer and raises when it finds none, which is right for the effects
benchmark, where the model has to return TV/TE/DE/IE. Here the expected output
is the LaTeX source itself. This module makes the same call, against the same
endpoint and with the same configuration, and returns the raw text.
"""

from __future__ import annotations

import re
import time

from openai import OpenAI

from ..llm import LLM_CONFIGS

# Models wrap the output in a markdown block despite instructions to the
# contrary. Stripping it here beats penalising the report for it.
_FENCE_PATTERN = re.compile(r"^\s*```(?:latex|tex)?\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)

# The earlier pattern, ``<<([A-Z0-9_]+)>>``, accepted only well formed tokens
# and so missed the very case this check exists for: when the template
# translation escaped the underscore inside a token (``<<PROTECTED\_ATTR>>``),
# nobody filled the placeholder, it shipped in the report, and this returned an
# empty list. A malformed token is an unfilled placeholder.
#
# The class takes letters, digits, underscore and the backslash of a bad
# escape, but not dots or spaces: the prompt itself contains the literal
# ``<<...>>`` as an example, and a wider class would report it as a slot.
_PLACEHOLDER_PATTERN = re.compile(r"<<([A-Za-z0-9_\\]+)>>")


def strip_markdown_fences(text: str) -> str:
    match = _FENCE_PATTERN.match(text.strip())
    return match.group(1).strip() if match else text.strip()


def extract_latex_document(text: str) -> str:
    """Isolate the document from ``\\documentclass`` to ``\\end{document}``.

    Discards any prose the model adds before or after the source, without
    touching the document itself.
    """
    cleaned = strip_markdown_fences(text)
    start = cleaned.find(r"\documentclass")
    end = cleaned.rfind(r"\end{document}")
    if start == -1 or end == -1:
        return cleaned
    return cleaned[start : end + len(r"\end{document}")]


def find_unfilled_placeholders(latex_text: str) -> list[str]:
    """Placeholders left in the document.

    A non-empty list means the model did not complete the template, and the
    report counts as malformed.
    """
    return _PLACEHOLDER_PATTERN.findall(latex_text)


def call_llm_report(
    system_prompt: str,
    user_prompt: str,
    config: dict | None = None,
    max_tokens: int = 4096,
    cache_prompt: bool = False,
    require_complete: bool = True,
) -> tuple[str, dict, float]:
    """Send both prompts and return ``(latex, usage, elapsed_seconds)``.

    The LaTeX comes back already stripped of markdown fences and of anything
    outside the document. It is not validated here; the validator does that.

    With ``require_complete``, the default, a truncated answer raises instead
    of being returned. Running out of ``max_tokens`` is a failure mode already
    seen with this model, and a truncated report would still produce a score,
    which is a number that looks like a result without being one. Turn the
    check off explicitly to inspect partial output.
    """
    if config is None:
        config = LLM_CONFIGS[0]

    client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
        # As in src.llm.call_llm: llama.cpp routes a request to the slot with
        # the longest common prefix and reuses its KV cache, so identical
        # requests can diverge even at temperature zero. Disabled, because a
        # report that cannot be reproduced cannot be assessed.
        extra_body={"cache_prompt": cache_prompt},
    )
    elapsed = time.perf_counter() - start

    finish_reason = response.choices[0].finish_reason
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "finish_reason": finish_reason,
    }

    if require_complete and finish_reason != "stop":
        raise ValueError(
            f"The model did not finish its answer "
            f"(finish_reason={finish_reason!r}, max_tokens={max_tokens}). "
            "The report is incomplete and must not be scored: raise max_tokens, "
            "or pass require_complete=False to inspect the partial output."
        )

    latex = extract_latex_document(response.choices[0].message.content)
    return latex, usage, elapsed
