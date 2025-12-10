# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Full System Integration Demo
=======================================

Demonstrates the synchronous execution of multi-modality pipelines
AND the generation of RICH OUTPUT DOSSIERS (>500 metrics possible).
"""

import sys
import os
import time

# Add root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rhenium.pipelines.runner import run_pipeline, PipelineRegistry
import rhenium.pipelines.xray_pipeline
import rhenium.pipelines.ct_pipeline
import rhenium.pipelines.us_pipeline


def print_dossier_summary(context, pipeline_name):
    print(f"\n[OUTPUT ORCHESTRATOR REPORT: {pipeline_name}]")
    if context.output_manager:
        print(context.output_manager.get_summary())
        # Print a few metrics to show depth
        study = context.output_manager.dossier.study_output
        print("\n--- Selected Metrics ---")
        all_metrics = study.global_metrics + [m for o in study.organs for m in o.metrics]
        for m in all_metrics[:5]: # Show first 5
            print(f" - {m.name}: {m.value:.2f} {m.unit}")
        if len(all_metrics) > 5:
            print(f" ... and {len(all_metrics) - 5} more metrics.")
            
        print("\n--- Generated Maps/Artifacts ---")
        for m in context.output_manager.dossier.maps:
            print(f" - [{m.type}] {m.file_path} ({m.dimensions})")
    else:
        print("No output manager attached.")
    print("-" * 50)


def main():
    print("="*60)
    print("      RHENIUM OS - OUTPUT EXPLOSION DEMO (v2025.12)      ")
    print("="*60)
    print("\nInitializing Rhenium OS Output Engine...\n")

    # 1. Run Chest X-ray Pipeline
    print("\n>> TASK 1: Chest X-ray Triage (Patient: A-101)")
    ctx_cxr = run_pipeline("cxr_pipeline", patient_id="A-101", study_uid="std-xray-001")
    print_dossier_summary(ctx_cxr, "Chest X-ray")
    
    # 2. Run CT Pipeline
    print("\n>> TASK 2: Lung Cancer Screening CT (Patient: B-202)")
    ctx_ct = run_pipeline("ct_pipeline", patient_id="B-202", study_uid="std-ct-002", metadata={"region": "chest"})
    print_dossier_summary(ctx_ct, "Chest CT")
    
    # 3. Run Ultrasound Pipeline
    print("\n>> TASK 3: Cardiac Echo Analysis (Patient: C-303)")
    ctx_us = run_pipeline("us_pipeline", patient_id="C-303", study_uid="std-us-003", metadata={"application": "cardiac"})
    print_dossier_summary(ctx_us, "Echo Ultrasound")
    
    print("\n" + "="*60)
    print("SYSTEM STATUS: RICH OUTPUT GENERATION ACTIVE.")
    print("Evidence Dossiers serialized to disk.")
    print("="*60)

if __name__ == "__main__":
    main()
