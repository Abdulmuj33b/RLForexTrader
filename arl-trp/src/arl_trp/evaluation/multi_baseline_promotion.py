from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from arl_trp.evaluation.comparison import (
    BaselineComparisonResult,
)


class MultiBaselinePromotionDecision(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class MultiBaselinePromotionConfig:
    """
    Promotion protocol for candidates evaluated against
    multiple mandatory baselines.

    v0.1:

        Performance superiority:
            candidate Sharpe must exceed every required
            performance baseline.

        Risk constraint:
            candidate maximum drawdown must not exceed
            each risk baseline by more than the configured
            degradation allowance.

        Flat/no-trade is a performance diagnostic baseline,
        but is not a risk baseline because its market
        exposure is zero.

    """

    folds_required: int = 5
    confidence_level: float = 0.95
    max_drawdown_degradation: float = 0.10

    required_baselines: tuple[str, ...] = (
        "flat",
        "buy_and_hold",
        "momentum",
        "random",
    )

    performance_baselines: tuple[str, ...] = (
        "flat",
        "buy_and_hold",
        "momentum",
        "random",
    )

    risk_baselines: tuple[str, ...] = (
        "buy_and_hold",
        "momentum",
        "random",
    )

    require_oos_superiority: bool = True
    require_baseline_comparison: bool = True
    require_dsr: bool = True
    require_simulator_validation: bool = True

    protocol_version: str = (
        "multi_baseline_promotion_v0.1"
    )

    def __post_init__(self) -> None:
        if self.folds_required <= 0:
            raise ValueError(
                "folds_required must be positive"
            )

        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                "confidence_level must be between 0 and 1"
            )

        if self.max_drawdown_degradation < 0.0:
            raise ValueError(
                "max_drawdown_degradation cannot be negative"
            )

        if not self.required_baselines:
            raise ValueError(
                "required_baselines cannot be empty"
            )

        if not self.performance_baselines:
            raise ValueError(
                "performance_baselines cannot be empty"
            )

        if not self.risk_baselines:
            raise ValueError(
                "risk_baselines cannot be empty"
            )

        required = set(self.required_baselines)

        if len(required) != len(self.required_baselines):
            raise ValueError(
                "required baseline names must be unique"
            )

        if len(set(self.performance_baselines)) != len(
            self.performance_baselines
        ):
            raise ValueError(
                "performance baseline names must be unique"
            )

        if len(set(self.risk_baselines)) != len(
            self.risk_baselines
        ):
            raise ValueError(
                "risk baseline names must be unique"
            )

        if not set(self.performance_baselines).issubset(
            required
        ):
            raise ValueError(
                "performance baselines must be required baselines"
            )

        if not set(self.risk_baselines).issubset(required):
            raise ValueError(
                "risk baselines must be required baselines"
            )

        if not self.protocol_version:
            raise ValueError(
                "protocol_version cannot be empty"
            )


@dataclass(frozen=True)
class MultiBaselinePromotionEvidence:
    """
    Complete evidence presented to the multi-baseline gate.
    """

    fold_count: int

    candidate_sharpe: float
    candidate_bootstrap_lower_bound: float
    candidate_max_drawdown: float

    comparison: BaselineComparisonResult

    baseline_max_drawdowns: dict[str, float]

    dsr_passed: bool
    simulator_validated: bool

    confidence_level: float

    def __post_init__(self) -> None:
        if self.fold_count < 0:
            raise ValueError(
                "fold_count cannot be negative"
            )

        if self.candidate_max_drawdown < 0.0:
            raise ValueError(
                "candidate_max_drawdown cannot be negative"
            )

        if not self.baseline_max_drawdowns:
            raise ValueError(
                "baseline_max_drawdowns cannot be empty"
            )

        for name, drawdown in (
            self.baseline_max_drawdowns.items()
        ):
            if not name:
                raise ValueError(
                    "baseline name cannot be empty"
                )

            if drawdown < 0.0:
                raise ValueError(
                    "baseline maximum drawdown cannot be negative"
                )

        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                "confidence_level must be between 0 and 1"
            )


@dataclass(frozen=True)
class MultiBaselinePromotionDecisionResult:
    """
    Auditable result of a multi-baseline promotion decision.
    """

    decision: MultiBaselinePromotionDecision
    protocol_version: str
    reasons: tuple[str, ...]

    fold_count: int

    candidate_sharpe: float
    candidate_bootstrap_lower_bound: float
    candidate_max_drawdown: float

    comparison: BaselineComparisonResult
    baseline_max_drawdowns: dict[str, float]

    dsr_passed: bool
    simulator_validated: bool

    confidence_level: float

    @property
    def passed(self) -> bool:
        return (
            self.decision
            == MultiBaselinePromotionDecision.PASS
        )


class MultiBaselinePromotionGate:
    """
    Explicit multi-baseline promotion gate.

    v0.1 promotion requirements:

    1. Required number of OOS folds.
    2. Every required baseline must be present.
    3. Candidate Sharpe must exceed every performance
       baseline when OOS superiority is required.
    4. Candidate drawdown must satisfy the degradation
       limit against every risk baseline.
    5. DSR must pass when required.
    6. Simulator validation must pass when required.

    No baseline is aggregated into a synthetic "worst"
    baseline. Each comparison remains independently
    auditable.
    """

    PROTOCOL_VERSION = (
        "multi_baseline_promotion_v0.1"
    )

    def __init__(
        self,
        config: MultiBaselinePromotionConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else MultiBaselinePromotionConfig()
        )

    def evaluate(
        self,
        evidence: MultiBaselinePromotionEvidence,
    ) -> MultiBaselinePromotionDecisionResult:
        reasons: list[str] = []

        config = self.config

        # --------------------------------------------------
        # 1. Required folds
        # --------------------------------------------------

        if evidence.fold_count < config.folds_required:
            reasons.append(
                "insufficient OOS folds: "
                f"{evidence.fold_count} < "
                f"{config.folds_required}"
            )

        # --------------------------------------------------
        # 2. Required baselines
        # --------------------------------------------------

        comparison_names = {
            comparison.baseline_name
            for comparison in evidence.comparison.comparisons
        }

        missing_baselines = (
            set(config.required_baselines)
            - comparison_names
        )

        if missing_baselines:
            reasons.append(
                "missing required baselines: "
                + ", ".join(
                    sorted(missing_baselines)
                )
            )

        missing_drawdowns = (
            set(config.required_baselines)
            - set(evidence.baseline_max_drawdowns)
        )

        if missing_drawdowns:
            reasons.append(
                "missing baseline drawdown evidence: "
                + ", ".join(
                    sorted(missing_drawdowns)
                )
            )

        # --------------------------------------------------
        # 3. Performance superiority
        # --------------------------------------------------

        comparisons_by_name = {
            comparison.baseline_name: comparison
            for comparison in evidence.comparison.comparisons
        }

        if (
            config.require_baseline_comparison
            and config.require_oos_superiority
        ):
            for baseline_name in (
                config.performance_baselines
            ):
                comparison = comparisons_by_name.get(
                    baseline_name
                )

                if comparison is None:
                    continue

                if not comparison.sharpe_superior:
                    reasons.append(
                        "candidate Sharpe does not exceed "
                        f"performance baseline "
                        f"'{baseline_name}': "
                        f"{comparison.candidate_sharpe} <= "
                        f"{comparison.baseline_sharpe}"
                    )

        # --------------------------------------------------
        # 4. Risk / drawdown constraint
        # --------------------------------------------------

        for baseline_name in config.risk_baselines:
            comparison = comparisons_by_name.get(
                baseline_name
            )

            if comparison is None:
                continue

            baseline_drawdown = (
                evidence.baseline_max_drawdowns.get(
                    baseline_name
                )
            )

            if baseline_drawdown is None:
                continue

            allowed_drawdown = (
                baseline_drawdown
                * (1.0 + config.max_drawdown_degradation)
            )

            if (
                evidence.candidate_max_drawdown
                > allowed_drawdown
            ):
                reasons.append(
                    "candidate maximum drawdown exceeds "
                    f"allowed degradation versus "
                    f"'{baseline_name}': "
                    f"{evidence.candidate_max_drawdown} > "
                    f"{allowed_drawdown}"
                )

        # --------------------------------------------------
        # 5. DSR
        # --------------------------------------------------

        if (
            config.require_dsr
            and not evidence.dsr_passed
        ):
            reasons.append(
                "deflated Sharpe ratio requirement failed"
            )

        # --------------------------------------------------
        # 6. Simulator validation
        # --------------------------------------------------

        if (
            config.require_simulator_validation
            and not evidence.simulator_validated
        ):
            reasons.append(
                "simulator validation requirement failed"
            )

        # --------------------------------------------------
        # Final decision
        # --------------------------------------------------

        decision = (
            MultiBaselinePromotionDecision.PASS
            if not reasons
            else MultiBaselinePromotionDecision.FAIL
        )

        return MultiBaselinePromotionDecisionResult(
            decision=decision,
            protocol_version=config.protocol_version,
            reasons=tuple(reasons),
            fold_count=evidence.fold_count,
            candidate_sharpe=evidence.candidate_sharpe,
            candidate_bootstrap_lower_bound=(
                evidence.candidate_bootstrap_lower_bound
            ),
            candidate_max_drawdown=(
                evidence.candidate_max_drawdown
            ),
            comparison=evidence.comparison,
            baseline_max_drawdowns=dict(
                evidence.baseline_max_drawdowns
            ),
            dsr_passed=evidence.dsr_passed,
            simulator_validated=evidence.simulator_validated,
            confidence_level=evidence.confidence_level,
        )