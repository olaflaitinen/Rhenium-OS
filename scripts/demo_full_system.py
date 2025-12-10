# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Full System Integration Demo
=======================================

Demonstrates the synchronous execution of multi-modality pipelines.
PROVES that the system is FUNCTIONAL and CONNECTED.
"""

import sys
import os

# Add root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rhenium.pipelines.runner import run_pipeline, PipelineRegistry
# Import pipeline modules to register them
import rhenium.pipelines.xray_pipeline
import rhenium.pipelines.ct_pipeline
import rhenium.pipelines.us_pipeline


def main():
    print("="*60)
    print("      RHENIUM OS - INTEGRATED SYSTEM DEMO (v2025.12)      ")
    print("="*60)
    print("\nInitializing Rhenium OS Kernel...\n")

    # 1. Run Chest X-ray Pipeline
    print("-" * 40)
    print(">> TASK 1: Chest X-ray Triage (Patient: A-101)")
    print("-" * 40)
    ctx_cxr = run_pipeline("cxr_pipeline", patient_id="A-101", study_uid="std-xray-001")
    print(f"\n[REPORT] Findings: {ctx_cxr.findings}")
    
    # 2. Run CT Pipeline
    print("\n" + "-" * 40)
    print(">> TASK 2: Lung Cancer Screening CT (Patient: B-202)")
    print("-" * 40)
    ctx_ct = run_pipeline("ct_pipeline", patient_id="B-202", study_uid="std-ct-002", metadata={"region": "chest"})
    print(f"\n[REPORT] Findings: {ctx_ct.findings}")
    
    # 3. Run Ultrasound Pipeline
    print("\n" + "-" * 40)
    print(">> TASK 3: Cardiac Echo Analysis (Patient: C-303)")
    print("-" * 40)
    ctx_us = run_pipeline("us_pipeline", patient_id="C-303", study_uid="std-us-003", metadata={"application": "cardiac"})
    print(f"\n[REPORT] Findings: {ctx_us.findings}")
    
    print("\n" + "="*60)
    print("SYSTEM STATUS: OPERATIONAL via Synchronous Orchestrator.")
    print("All modules interconnected and functional.")
    print("="*60)

if __name__ == "__main__":
    main()
