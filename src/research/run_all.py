"""Run the complete Step-1500 research suite in dependency order."""

from __future__ import annotations

import argparse
from typing import Sequence

from src.research.calibrate_probabilities import calibrate_probabilities
from src.research.error_analysis import run_error_analysis
from src.research.evaluate_topk import evaluate_top_k
from src.research.feature_ablation import run_feature_ablation
from src.research.model_disagreement import run_model_disagreement
from src.research.robustness_analysis import run_robustness_analysis
from src.research.shap_explain import run_shap_analysis
from src.research.summarize_research import generate_research_summary
from src.research.tune_random_forest import tune_random_forest


def run_all_research(*, quick_tuning: bool = False) -> None:
    """Run every experiment; quick tuning is intended only for development checks."""

    evaluate_top_k(verbose=True)
    calibrate_probabilities(verbose=True)
    run_shap_analysis(verbose=True)
    tune_random_forest(quick=quick_tuning, verbose=True)
    run_feature_ablation(verbose=True)
    run_robustness_analysis(verbose=True)
    run_model_disagreement(verbose=True)
    run_error_analysis(verbose=True)
    generate_research_summary(verbose=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick-tuning",
        action="store_true",
        help="Use two tuning candidates for development only, not paper artifacts.",
    )
    args = parser.parse_args(argv)
    run_all_research(quick_tuning=args.quick_tuning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
