from dataclasses import dataclass
from enum import Enum


class PromotionDecision(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class PromotionConfig:
    """
    Frozen v0.1 champion/challenger promotion protocol.

    Promotion never depends on profitability alone. Every required
    criterion must pass.

    v0.1 protocol:
        folds_required = 5
        confidence_level = 0.95
        max_drawdown_degradation = 0.10
        require_oos_superiority = True
        require_baseline_comparison = True
        require_dsr = True
        require_simulator_validation = True
    """

    folds_required: int = 5
    confidence_level: float = 0.95
    max_drawdown_degradation: float = 0.10

    require_oos_superiority: bool = True
    require_baseline_comparison: bool = True
    require_dsr: bool = True
    require_simulator_validation: bool = True

    protocol_version: str = "promotion_v0.1"

    def __post_init__(self) -> None:
        if self.folds_required <= 0:
            raise ValueError("folds_required must be positive")

        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                "confidence_level must be between 0 and 1"
            )

        if self.max_drawdown_degradation < 0.0:
            raise ValueError(
                "max_drawdown_degradation cannot be negative"
            )

        if not self.protocol_version:
            raise ValueError("protocol_version cannot be empty")


@dataclass(frozen=True)
class PromotionEvidence:
    """
    Evidence supplied to the promotion gate.

    The gate evaluates evidence; it does not calculate performance
    statistics itself.
    """

    fold_count: int

    candidate_sharpe: float
    candidate_bootstrap_lower_bound: float
    candidate_max_drawdown: float

    baseline_sharpe: float | None
    baseline_max_drawdown: float | None

    dsr_passed: bool
    simulator_validated: bool

    confidence_level: float

    def __post_init__(self) -> None:
        if self.fold_count < 0:
            raise ValueError("fold_count cannot be negative")

        if self.candidate_max_drawdown < 0.0:
            raise ValueError(
                "candidate_max_drawdown cannot be negative"
            )

        if self.baseline_max_drawdown is not None:
            if self.baseline_max_drawdown < 0.0:
                raise ValueError(
                    "baseline_max_drawdown cannot be negative"
                )

        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                "confidence_level must be between 0 and 1"
            )


@dataclass(frozen=True)
class PromotionDecisionResult:
    """
    Auditable result of a champion/challenger promotion decision.
    """

    decision: PromotionDecision
    protocol_version: str
    reasons: tuple[str, ...]

    fold_count: int
    candidate_sharpe: float
    candidate_bootstrap_lower_bound: float
    candidate_max_drawdown: float

    baseline_sharpe: float | None
    baseline_max_drawdown: float | None

    dsr_passed: bool
    simulator_validated: bool

    @property
    def passed(self) -> bool:
        return self.decision == PromotionDecision.PASS


class PromotionGate:
    """
    Deterministic champion/challenger promotion gate.

    v0.1 rules:

    1. Required number of OOS folds must be available.
    2. Candidate must outperform the baseline on pooled OOS Sharpe
       when OOS superiority is required.
    3. Candidate maximum drawdown may not exceed the baseline by more
       than the configured degradation allowance.
    4. DSR must pass when required.
    5. Simulator validation must pass when required.

    The bootstrap confidence interval is retained as evidence but is
    not independently converted into a new promotion threshold here.
    This avoids silently adding an unfrozen statistical criterion.
    """

    def __init__(self, config: PromotionConfig | None = None) -> None:
        self.config = config or PromotionConfig()

    def evaluate(
        self,
        evidence: PromotionEvidence,
    ) -> PromotionDecisionResult:
        reasons: list[str] = []
        passed = True

        if evidence.fold_count < self.config.folds_required:
            passed = False
            reasons.append(
                "required OOS fold count was not reached"
            )
        else:
            reasons.append(
                "required OOS fold count was reached"
            )

        if self.config.require_baseline_comparison:
            if (
                evidence.baseline_sharpe is None
                or evidence.baseline_max_drawdown is None
            ):
                passed = False
                reasons.append(
                    "baseline comparison evidence is missing"
                )
            else:
                reasons.append(
                    "baseline comparison evidence is present"
                )

                if self.config.require_oos_superiority:
                    if evidence.candidate_sharpe <= evidence.baseline_sharpe:
                        passed = False
                        reasons.append(
                            "candidate does not exceed baseline OOS Sharpe"
                        )
                    else:
                        reasons.append(
                            "candidate exceeds baseline OOS Sharpe"
                        )

                allowed_drawdown = (
                    evidence.baseline_max_drawdown
                    * (1.0 + self.config.max_drawdown_degradation)
                )

                if evidence.candidate_max_drawdown > allowed_drawdown:
                    passed = False
                    reasons.append(
                        "candidate maximum drawdown exceeds "
                        "the permitted baseline degradation"
                    )
                else:
                    reasons.append(
                        "candidate maximum drawdown is within "
                        "the permitted baseline degradation"
                    )

        elif self.config.require_oos_superiority:
            passed = False
            reasons.append(
                "OOS superiority requires baseline comparison"
            )

        if self.config.require_dsr:
            if not evidence.dsr_passed:
                passed = False
                reasons.append("DSR requirement failed")
            else:
                reasons.append("DSR requirement passed")

        if self.config.require_simulator_validation:
            if not evidence.simulator_validated:
                passed = False
                reasons.append(
                    "simulator validation requirement failed"
                )
            else:
                reasons.append(
                    "simulator validation requirement passed"
                )

        decision = (
            PromotionDecision.PASS
            if passed
            else PromotionDecision.FAIL
        )

        return PromotionDecisionResult(
            decision=decision,
            protocol_version=self.config.protocol_version,
            reasons=tuple(reasons),
            fold_count=evidence.fold_count,
            candidate_sharpe=evidence.candidate_sharpe,
            candidate_bootstrap_lower_bound=(
                evidence.candidate_bootstrap_lower_bound
            ),
            candidate_max_drawdown=evidence.candidate_max_drawdown,
            baseline_sharpe=evidence.baseline_sharpe,
            baseline_max_drawdown=evidence.baseline_max_drawdown,
            dsr_passed=evidence.dsr_passed,
            simulator_validated=evidence.simulator_validated,
        )
