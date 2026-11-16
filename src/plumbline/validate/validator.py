"""Output validation: nothing a model says reaches the user unverified.

The validator turns a raw model reply into one of three verdicts:

  pass      every claim is cited, quoted and supported
  degraded  some claims failed and were removed; the surviving ones are served
  rejected  nothing trustworthy is left (unparseable, off-schema, or wholly unsupported)

`rejected` comes with machine-readable issues and a plain-language feedback message that
the resilience layer can send back to the model for one repair attempt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from plumbline.llm.base import Passage
from plumbline.validate import grounding
from plumbline.validate.parse import ParseFailure, extract_json
from plumbline.validate.schema import Answer, Claim


@dataclass
class Issue:
    code: str
    message: str
    claim: int | None = None

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "claim": self.claim}


@dataclass
class ValidationReport:
    status: str  # pass | degraded | rejected
    answer: Answer | None = None
    issues: list[Issue] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    claims_in: int = 0
    claims_kept: int = 0

    @property
    def ok(self) -> bool:
        return self.status in ("pass", "degraded")

    def feedback(self) -> str:
        return "\n".join(f"- {i.message}" for i in self.issues) or "- the reply was not usable"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "issues": [i.to_dict() for i in self.issues],
            "checks": self.checks,
            "claims_in": self.claims_in,
            "claims_kept": self.claims_kept,
        }


def _check(report: ValidationReport, name: str, passed: bool, detail: str = "") -> None:
    report.checks.append({"name": name, "passed": passed, "detail": detail})


def validate_output(raw: str, passages: list[Passage], question: str = "") -> ValidationReport:
    report = ValidationReport(status="rejected")

    # 1. is it JSON at all?
    try:
        data = extract_json(raw)
    except ParseFailure as exc:
        report.issues.append(Issue(exc.code, f"Could not read the reply as JSON: {exc}"))
        _check(report, "json", False, str(exc))
        return report
    _check(report, "json", True)

    # 2. does it follow the contract?
    try:
        answer = Answer.model_validate(data)
    except ValidationError as exc:
        for err in exc.errors()[:4]:
            where = ".".join(str(p) for p in err["loc"]) or "object"
            report.issues.append(Issue("schema", f"Field `{where}`: {err['msg']}"))
        _check(report, "schema", False, f"{exc.error_count()} schema error(s)")
        return report
    _check(report, "schema", True)
    report.claims_in = len(answer.claims)

    # 3. an abstention is a valid, complete answer
    if answer.abstained:
        report.status = "pass"
        report.answer = Answer(abstained=True, answer="", claims=[], confidence=answer.confidence)
        _check(report, "abstention", True, "model abstained")
        return report

    # 4. every claim must survive citation, quote, support and number checks
    by_id = {p.id: p for p in passages}
    kept: list[Claim] = []
    tallies = {"citations": 0, "quote": 0, "support": 0, "numbers": 0}
    for i, claim in enumerate(answer.claims):
        known = [c for c in claim.citations if c in by_id]
        unknown = [c for c in claim.citations if c not in by_id]
        if unknown or not known:
            what = f"cites {unknown} which were not among the passages" if unknown else "has no citation"
            report.issues.append(Issue("unknown_citation", f"Claim {i + 1} {what}.", i))
            tallies["citations"] += 1
            continue
        cited = [by_id[c].text for c in known]

        quote_tokens = len(claim.quote.split())
        best = max((grounding.quote_match(claim.quote, t) for t in cited), default=0.0)
        if quote_tokens < min(grounding.QUOTE_MIN_TOKENS, len(claim.text.split())) or (
            best < grounding.QUOTE_FUZZY_RATIO
        ):
            report.issues.append(Issue(
                "quote_not_found",
                f"Claim {i + 1}: the quote does not appear in the cited passage"
                f" (best contiguous match {best:.0%}).", i))
            tallies["quote"] += 1
            continue

        coverage = grounding.support_coverage(claim.text, cited)
        if coverage < grounding.SUPPORT_MIN_COVERAGE:
            report.issues.append(Issue(
                "unsupported_claim",
                f"Claim {i + 1}: only {coverage:.0%} of its content is found in the cited passage.", i))
            tallies["support"] += 1
            continue

        stray = grounding.unsupported_numbers(claim.text, cited, question)
        if stray:
            report.issues.append(Issue(
                "ungrounded_number",
                f"Claim {i + 1}: the figure(s) {sorted(stray)} are not in the cited passage.", i))
            tallies["numbers"] += 1
            continue
        kept.append(Claim(text=claim.text, citations=known, quote=claim.quote))

    n = len(answer.claims)
    _check(report, "citations_exist", tallies["citations"] == 0, f"{tallies['citations']}/{n} failed")
    _check(report, "quote_verbatim", tallies["quote"] == 0, f"{tallies['quote']}/{n} failed")
    _check(report, "claim_supported", tallies["support"] == 0, f"{tallies['support']}/{n} failed")
    _check(report, "numbers_grounded", tallies["numbers"] == 0, f"{tallies['numbers']}/{n} failed")
    report.claims_kept = len(kept)

    if not kept:
        if not answer.claims:
            report.issues.append(Issue("no_claims", "The answer is not broken down into cited claims."))
        _check(report, "has_supported_claim", False)
        return report
    _check(report, "has_supported_claim", True)

    if len(kept) == n:
        report.status = "pass"
        report.answer = Answer(abstained=False, answer=answer.answer or _join(kept),
                               claims=kept, confidence=answer.confidence)
    else:
        # serve only what was verified: the free-text answer may still contain a dropped
        # claim, so it is rebuilt from the surviving ones
        report.status = "degraded"
        report.answer = Answer(abstained=False, answer=_join(kept), claims=kept,
                               confidence=answer.confidence * len(kept) / n)
    return report


def _join(claims: list[Claim]) -> str:
    return " ".join(c.text.rstrip() for c in claims)
