"""Independent, rule-based soil nutrient assessment.

No agronomic cut-offs are built into this module.  The safe default is
``"Needs verification"`` for every status.  Classification is enabled only
when callers provide a source for each rule and explicitly mark that rule as
verified.  This keeps provisional values from being presented as scientific
guidance and keeps these rules separate from the ML crop prediction.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np


SOIL_FEATURES = ("N", "P", "K", "ph")
NEEDS_VERIFICATION = "Needs verification"
DEFAULT_THRESHOLD_SOURCE = "Not configured; scientific verification required."


@dataclass(frozen=True)
class ThresholdRule:
    """A sourced lower/upper interval for one soil indicator.

    Values below ``lower`` are classified Low, values from ``lower`` through
    ``upper`` are Adequate (Suitable for pH), and values above ``upper`` are
    High.  ``verified`` must be explicitly true before those labels are used.
    """

    lower: float
    upper: float
    source: str
    verified: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.lower, bool) or not isinstance(self.lower, Real):
            raise TypeError("Threshold lower bound must be numeric.")
        if isinstance(self.upper, bool) or not isinstance(self.upper, Real):
            raise TypeError("Threshold upper bound must be numeric.")
        if not math.isfinite(float(self.lower)) or not math.isfinite(float(self.upper)):
            raise ValueError("Threshold bounds must be finite.")
        if float(self.lower) > float(self.upper):
            raise ValueError("Threshold lower bound cannot exceed upper bound.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Every configured threshold must document its source.")
        if not isinstance(self.verified, bool):
            raise TypeError("Threshold verified flag must be boolean.")


def _number(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a numeric value, not {type(value).__name__}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def validate_soil_inputs(N: Any, P: Any, K: Any, ph: Any) -> dict[str, float]:
    """Validate the four soil inputs without applying suitability cut-offs."""

    values = {
        "N": _number("N", N),
        "P": _number("P", P),
        "K": _number("K", K),
        "ph": _number("ph", ph),
    }
    for name in ("N", "P", "K"):
        if values[name] < 0:
            raise ValueError(f"{name} cannot be negative.")
    if not 0 <= values["ph"] <= 14:
        raise ValueError("ph must be between 0 and 14.")
    return values


def _rule_from_value(
    feature: str,
    value: ThresholdRule | Mapping[str, Any],
    shared_source: str | None,
) -> ThresholdRule:
    if isinstance(value, ThresholdRule):
        rule = value
    elif isinstance(value, Mapping):
        allowed = {
            "lower",
            "upper",
            "lower_bound",
            "upper_bound",
            "source",
            "verified",
        }
        unexpected = sorted(str(key) for key in set(value) - allowed)
        if unexpected:
            raise ValueError(
                f"Unexpected threshold fields for {feature}: {', '.join(unexpected)}."
            )
        lower = value.get("lower", value.get("lower_bound"))
        upper = value.get("upper", value.get("upper_bound"))
        if lower is None or upper is None:
            raise ValueError(
                f"Threshold for {feature} requires lower and upper bounds."
            )
        source = value.get("source", shared_source)
        if source is None:
            raise ValueError(f"Threshold for {feature} requires a documented source.")
        rule = ThresholdRule(
            lower=lower,
            upper=upper,
            source=source,
            verified=value.get("verified", False),
        )
    else:
        raise TypeError(
            f"Threshold for {feature} must be ThresholdRule or a mapping."
        )

    lower = float(rule.lower)
    upper = float(rule.upper)
    if feature in {"N", "P", "K"} and lower < 0:
        raise ValueError(f"{feature} threshold bounds cannot be negative.")
    if feature == "ph" and not (0 <= lower <= upper <= 14):
        raise ValueError("ph threshold bounds must be between 0 and 14.")
    return rule


def _normalise_thresholds(
    thresholds: Mapping[str, ThresholdRule | Mapping[str, Any]] | None,
    threshold_source: str | None,
) -> dict[str, ThresholdRule]:
    if thresholds is None:
        return {}
    if not isinstance(thresholds, Mapping):
        raise TypeError("thresholds must be a mapping keyed by N, P, K, and ph.")
    unexpected = sorted(str(key) for key in set(thresholds) - set(SOIL_FEATURES))
    if unexpected:
        raise ValueError(f"Unexpected soil threshold keys: {', '.join(unexpected)}.")
    return {
        feature: _rule_from_value(feature, value, threshold_source)
        for feature, value in thresholds.items()
    }


def _status(feature: str, value: float, rule: ThresholdRule | None) -> str:
    if rule is None or not rule.verified:
        return NEEDS_VERIFICATION
    if value < float(rule.lower):
        return "Low"
    if value > float(rule.upper):
        return "High"
    return "Suitable" if feature == "ph" else "Adequate"


def assess_soil(
    N: Any,
    P: Any,
    K: Any,
    ph: Any,
    *,
    thresholds: Mapping[str, ThresholdRule | Mapping[str, Any]] | None = None,
    threshold_source: str | None = None,
) -> dict[str, Any]:
    """Return soil statuses independently of the machine-learning prediction.

    ``thresholds`` may contain any subset of ``N``, ``P``, ``K``, and ``ph``.
    Each entry needs numeric ``lower``/``upper`` bounds, a non-empty ``source``
    (or shared ``threshold_source``), and ``verified=True`` to activate the
    Low/Adequate-or-Suitable/High labels.  Missing or provisional rules retain
    the safe ``Needs verification`` status.
    """

    values = validate_soil_inputs(N, P, K, ph)
    rules = _normalise_thresholds(thresholds, threshold_source)
    statuses = {
        feature: _status(feature, values[feature], rules.get(feature))
        for feature in SOIL_FEATURES
    }

    if any(status == NEEDS_VERIFICATION for status in statuses.values()):
        overall = NEEDS_VERIFICATION
    elif (
        statuses["N"] == "Adequate"
        and statuses["P"] == "Adequate"
        and statuses["K"] == "Adequate"
        and statuses["ph"] == "Suitable"
    ):
        overall = "Suitable under the configured sourced thresholds"
    else:
        overall = "One or more indicators are outside the configured sourced ranges"

    source_by_feature = {
        feature: rules[feature].source if feature in rules else DEFAULT_THRESHOLD_SOURCE
        for feature in SOIL_FEATURES
    }
    unique_sources = list(dict.fromkeys(source_by_feature.values()))
    all_verified = all(
        feature in rules and rules[feature].verified for feature in SOIL_FEATURES
    )

    return {
        "nitrogen_status": statuses["N"],
        "phosphorus_status": statuses["P"],
        "potassium_status": statuses["K"],
        "ph_status": statuses["ph"],
        "overall_assessment": overall,
        "thresholds_verified": all_verified,
        "threshold_source": "; ".join(unique_sources),
        "threshold_sources": source_by_feature,
        "verification_notice": (
            "All configured rules were explicitly marked verified by the caller."
            if all_verified
            else "One or more soil thresholds need verification from a credible "
            "agricultural reference."
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate soil inputs and apply optional sourced threshold rules."
    )
    parser.add_argument("N", type=float)
    parser.add_argument("P", type=float)
    parser.add_argument("K", type=float)
    parser.add_argument("ph", type=float)
    parser.add_argument(
        "--thresholds-json",
        type=Path,
        help="Optional JSON mapping of sourced lower/upper threshold rules.",
    )
    parser.add_argument(
        "--threshold-source",
        help="Shared source used by JSON rules that do not contain source.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    thresholds = None
    if args.thresholds_json is not None:
        with args.thresholds_json.open(encoding="utf-8") as handle:
            thresholds = json.load(handle)
    result = assess_soil(
        args.N,
        args.P,
        args.K,
        args.ph,
        thresholds=thresholds,
        threshold_source=args.threshold_source,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
