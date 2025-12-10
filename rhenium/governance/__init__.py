# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
Governance Module
=================

Audit, model cards, fairness, compliance, and risk management for
regulatory-ready medical imaging AI.

Last Updated: December 2025
"""

from rhenium.governance.audit_log import AuditLogger, log_pipeline_run, AuditEntry
from rhenium.governance.model_card import ModelCard
from rhenium.governance.risk_tracking import RiskTracker, RiskEntry, RiskSeverity
from rhenium.governance.data_governance import AccessLevel, DataAccessPolicy
from rhenium.governance.fairness_metrics import (
    FairnessMetrics,
    SubgroupMetrics,
    evaluate_fairness,
    compute_stratified_auc,
    compute_disparity_metrics,
)
from rhenium.governance.fairness_reports import (
    generate_fairness_report,
    assess_fairness_thresholds,
)
from rhenium.governance.bias_mitigation import (
    BiasMitigationStrategy,
    ReweightingStrategy,
    OversamplingStrategy,
    ThresholdAdjustmentStrategy,
    get_mitigation_strategy,
    MitigationConfig,
)

__all__ = [
    # Audit
    "AuditLogger",
    "AuditEntry",
    "log_pipeline_run",
    # Model documentation
    "ModelCard",
    # Risk management
    "RiskTracker",
    "RiskEntry",
    "RiskSeverity",
    # Data governance
    "AccessLevel",
    "DataAccessPolicy",
    # Fairness
    "FairnessMetrics",
    "SubgroupMetrics",
    "evaluate_fairness",
    "compute_stratified_auc",
    "compute_disparity_metrics",
    "generate_fairness_report",
    "assess_fairness_thresholds",
    # Bias mitigation
    "BiasMitigationStrategy",
    "ReweightingStrategy",
    "OversamplingStrategy",
    "ThresholdAdjustmentStrategy",
    "get_mitigation_strategy",
    "MitigationConfig",
]
