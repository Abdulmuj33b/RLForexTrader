import pytest

from arl_trp.evaluation.promotion import (
    PromotionConfig,
    PromotionDecision,
    PromotionEvidence,
    PromotionGate,
)


def make_passing_evidence() -> PromotionEvidence:
    return PromotionEvidence(
        fold_count=5,
        candidate_sharpe=1.20,
        candidate_bootstrap_lower_bound=0.20,
        candidate_max_drawdown=0.10,
        baseline_sharpe=0.80,
        baseline_max_drawdown=0.10,
        dsr_passed=True,
        simulator_validated=True,
        confidence_level=0.95,
    )


def test_passing_candidate_is_promoted():
    gate = PromotionGate()

    result = gate.evaluate(make_passing_evidence())

    assert result.decision == PromotionDecision.PASS
    assert result.passed is True
    assert result.protocol_version == "promotion_v0.1"


def test_insufficient_folds_fail():
    gate = PromotionGate()

    evidence = PromotionEvidence(
        **{
            **make_passing_evidence().__dict__,
            "fold_count": 4,
        }
    )

    result = gate.evaluate(evidence)

    assert result.decision == PromotionDecision.FAIL
    assert any(
        "fold count" in reason
        for reason in result.reasons
    )


def test_candidate_must_exceed_baseline_sharpe():
    gate = PromotionGate()

    evidence = PromotionEvidence(
        **{
            **make_passing_evidence().__dict__,
            "candidate_sharpe": 0.80,
            "baseline_sharpe": 0.80,
        }
    )

    result = gate.evaluate(evidence)

    assert result.decision == PromotionDecision.FAIL
    assert any(
        "does not exceed baseline OOS Sharpe" in reason
        for reason in result.reasons
    )


def test_drawdown_degradation_limit_is_enforced():
    config = PromotionConfig(
        max_drawdown_degradation=0.10,
    )
    gate = PromotionGate(config)

    evidence = PromotionEvidence(
        **{
            **make_passing_evidence().__dict__,
            "candidate_max_drawdown": 0.111,
            "baseline_max_drawdown": 0.10,
        }
    )

    result = gate.evaluate(evidence)

    assert result.decision == PromotionDecision.FAIL
    assert any(
        "maximum drawdown exceeds" in reason
        for reason in result.reasons
    )


def test_drawdown_within_allowed_degradation_passes():
    config = PromotionConfig(
        max_drawdown_degradation=0.10,
    )
    gate = PromotionGate(config)

    evidence = PromotionEvidence(
        **{
            **make_passing_evidence().__dict__,
            "candidate_max_drawdown": 0.11,
            "baseline_max_drawdown": 0.10,
        }
    )

    result = gate.evaluate(evidence)

    assert result.decision == PromotionDecision.PASS


def test_missing_baseline_fails_when_required():
    gate = PromotionGate()

    evidence = PromotionEvidence(
        **{
            **make_passing_evidence().__dict__,
            "baseline_sharpe": None,
            "baseline_max_drawdown": None,
        }
    )

    result = gate.evaluate(evidence)

    assert result.decision == PromotionDecision.FAIL
    assert any(
        "baseline comparison evidence is missing" in reason
        for reason in result.reasons
    )


def test_dsr_failure_blocks_promotion():
    gate = PromotionGate()

    evidence = PromotionEvidence(
        **{
            **make_passing_evidence().__dict__,
            "dsr_passed": False,
        }
    )

    result = gate.evaluate(evidence)

    assert result.decision == PromotionDecision.FAIL
    assert "DSR requirement failed" in result.reasons


def test_simulator_failure_blocks_promotion():
    gate = PromotionGate()

    evidence = PromotionEvidence(
        **{
            **make_passing_evidence().__dict__,
            "simulator_validated": False,
        }
    )

    result = gate.evaluate(evidence)

    assert result.decision == PromotionDecision.FAIL
    assert (
        "simulator validation requirement failed"
        in result.reasons
    )


def test_multiple_failures_are_preserved():
    gate = PromotionGate()

    evidence = PromotionEvidence(
        **{
            **make_passing_evidence().__dict__,
            "fold_count": 3,
            "candidate_sharpe": 0.20,
            "baseline_sharpe": 0.80,
            "candidate_max_drawdown": 0.30,
            "baseline_max_drawdown": 0.10,
            "dsr_passed": False,
            "simulator_validated": False,
        }
    )

    result = gate.evaluate(evidence)

    assert result.decision == PromotionDecision.FAIL

    assert any(
        "fold count" in reason
        for reason in result.reasons
    )
    assert any(
        "does not exceed baseline OOS Sharpe" in reason
        for reason in result.reasons
    )
    assert any(
        "maximum drawdown exceeds" in reason
        for reason in result.reasons
    )
    assert "DSR requirement failed" in result.reasons
    assert (
        "simulator validation requirement failed"
        in result.reasons
    )


def test_configuration_rejects_invalid_values():
    with pytest.raises(ValueError):
        PromotionConfig(folds_required=0)

    with pytest.raises(ValueError):
        PromotionConfig(confidence_level=1.0)

    with pytest.raises(ValueError):
        PromotionConfig(confidence_level=0.0)

    with pytest.raises(ValueError):
        PromotionConfig(max_drawdown_degradation=-0.01)

    with pytest.raises(ValueError):
        PromotionConfig(protocol_version="")


def test_evidence_rejects_invalid_values():
    passing = make_passing_evidence()

    with pytest.raises(ValueError):
        PromotionEvidence(
            **{
                **passing.__dict__,
                "fold_count": -1,
            }
        )

    with pytest.raises(ValueError):
        PromotionEvidence(
            **{
                **passing.__dict__,
                "candidate_max_drawdown": -0.01,
            }
        )

    with pytest.raises(ValueError):
        PromotionEvidence(
            **{
                **passing.__dict__,
                "baseline_max_drawdown": -0.01,
            }
        )

    with pytest.raises(ValueError):
        PromotionEvidence(
            **{
                **passing.__dict__,
                "confidence_level": 1.0,
            }
        )
