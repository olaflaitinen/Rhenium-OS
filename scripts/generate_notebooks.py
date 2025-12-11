"""
Rhenium OS Notebook Generator - Exact Structure
"""

import json
from pathlib import Path

# Exact notebook definitions from user specification
NOTEBOOKS = [
    # 00_index (01-02)
    (1, "00_index", "01_index_overview", "Global Index and Map"),
    (2, "00_index", "02_quickstart_clinical_and_ml_overview", "Quickstart Clinical and ML Overview"),
    
    # 10_architecture (03-10)
    (3, "10_architecture", "03_architecture_rhenium_os_overview", "System Architecture Overview"),
    (4, "10_architecture", "04_data_abstractions_and_types", "Data Abstractions and Types"),
    (5, "10_architecture", "05_pipeline_configuration_and_orchestration", "Pipeline Configuration and Orchestration"),
    (6, "10_architecture", "06_disease_reasoning_presence_and_identity", "Disease Reasoning: Presence and Identity"),
    (7, "10_architecture", "07_disease_reasoning_subtype_and_stage", "Disease Reasoning: Subtype and Stage"),
    (8, "10_architecture", "08_evidence_dossier_structure_and_save_load", "Evidence Dossier Structure"),
    (9, "10_architecture", "09_medgemma_integration_and_narratives", "MedGemma Integration and Narratives"),
    (10, "10_architecture", "10_cli_and_python_api_bridges", "CLI and Python API Bridges"),
    
    # 20_modality_mri (11-18)
    (11, "20_modality_mri", "11_mri_data_loading_and_preprocessing", "MRI Data Loading and Preprocessing"),
    (12, "20_modality_mri", "12_mri_brain_tumor_segmentation_pipeline", "MRI Brain Tumor Segmentation Pipeline"),
    (13, "20_modality_mri", "13_mri_ms_lesion_detection_pipeline", "MRI MS Lesion Detection Pipeline"),
    (14, "20_modality_mri", "14_mri_spine_degeneration_assessment", "MRI Spine Degeneration Assessment"),
    (15, "20_modality_mri", "15_mri_knee_msk_pipeline", "MRI Knee MSK Pipeline"),
    (16, "20_modality_mri", "16_mri_multi_sequence_fusion_and_features", "Multi-sequence MRI Fusion and Features"),
    (17, "20_modality_mri", "17_mri_reconstruction_and_denoising", "MRI Reconstruction and Denoising"),
    (18, "20_modality_mri", "18_mri_evidence_dossier_multistudy_example", "MRI Evidence Dossier Multi-study Example"),
    
    # 21_modality_ct (19-26)
    (19, "21_modality_ct", "19_ct_data_loading_and_windowing", "CT Data Loading and Windowing"),
    (20, "21_modality_ct", "20_ct_lung_nodule_detection_pipeline", "CT Lung Nodule Detection Pipeline"),
    (21, "21_modality_ct", "21_ct_pe_support_pipeline", "CT PE Support Pipeline"),
    (22, "21_modality_ct", "22_ct_liver_lesion_staging_surrogates", "CT Liver Lesion Staging Surrogates"),
    (23, "21_modality_ct", "23_ct_abdomen_critical_findings_pipeline", "CT Abdomen Critical Findings Pipeline"),
    (24, "21_modality_ct", "24_ct_spectral_features_and_maps", "CT Spectral Features and Maps"),
    (25, "21_modality_ct", "25_ct_perfusion_parametric_maps", "CT Perfusion Parametric Maps"),
    (26, "21_modality_ct", "26_ct_tumor_response_over_time", "CT Tumor Response Over Time"),
    
    # 22_modality_ultrasound (27-32)
    (27, "22_modality_ultrasound", "27_ultrasound_data_loading_and_preprocessing", "Ultrasound Data Loading and Preprocessing"),
    (28, "22_modality_ultrasound", "28_echo_lv_function_and_segmentation", "Echo LV Function and Segmentation"),
    (29, "22_modality_ultrasound", "29_ultrasound_abdominal_support_pipeline", "Ultrasound Abdominal Support Pipeline"),
    (30, "22_modality_ultrasound", "30_ultrasound_vascular_doppler_pipeline", "Ultrasound Vascular Doppler Pipeline"),
    (31, "22_modality_ultrasound", "31_ultrasound_elastography_stiffness_maps", "Ultrasound Elastography Stiffness Maps"),
    (32, "22_modality_ultrasound", "32_pocus_focused_assessment_pipeline", "POCUS Focused Assessment Pipeline"),
    
    # 23_modality_xray (33-38)
    (33, "23_modality_xray", "33_xray_data_loading_and_metadata", "X-ray Data Loading and Metadata"),
    (34, "23_modality_xray", "34_chest_xray_triage_and_xai", "Chest X-ray Triage and XAI"),
    (35, "23_modality_xray", "35_msk_xray_fracture_detection", "MSK X-ray Fracture Detection"),
    (36, "23_modality_xray", "36_mammography_lesion_support_pipeline", "Mammography Lesion Support Pipeline"),
    (37, "23_modality_xray", "37_dental_xray_tooth_and_caries_support", "Dental X-ray Tooth and Caries Support"),
    (38, "23_modality_xray", "38_xray_evidence_dossier_example", "X-ray Evidence Dossier Example"),
    
    # 30_disease_use_cases (39-61)
    (39, "30_disease_use_cases", "39_neuro_stroke_ct_mri_workflow", "Neuro Stroke CT/MRI Workflow"),
    (40, "30_disease_use_cases", "40_neuro_ich_detection_and_flags", "Neuro ICH Detection and Flags"),
    (41, "30_disease_use_cases", "41_neuro_brain_tumor_disease_reasoning", "Neuro Brain Tumor Disease Reasoning"),
    (42, "30_disease_use_cases", "42_neuro_neurodegeneration_volumetry", "Neuro Neurodegeneration Volumetry"),
    (43, "30_disease_use_cases", "43_neuro_ms_trajectory_analysis", "Neuro MS Trajectory Analysis"),
    (44, "30_disease_use_cases", "44_cardio_function_echo_mri_integration", "Cardio Function Echo MRI Integration"),
    (45, "30_disease_use_cases", "45_cardio_ct_coronary_support", "Cardio CT Coronary Support"),
    (46, "30_disease_use_cases", "46_cardio_pulmonary_hypertension_surrogates", "Cardio Pulmonary Hypertension Surrogates"),
    (47, "30_disease_use_cases", "47_cardio_heart_failure_phenotyping", "Cardio Heart Failure Phenotyping"),
    (48, "30_disease_use_cases", "48_cardio_safety_flags_and_trajectory", "Cardio Safety Flags and Trajectory"),
    (49, "30_disease_use_cases", "49_onco_lung_cancer_staging_surrogates", "Onco Lung Cancer Staging Surrogates"),
    (50, "30_disease_use_cases", "50_onco_liver_cancer_trajectory", "Onco Liver Cancer Trajectory"),
    (51, "30_disease_use_cases", "51_onco_breast_cancer_multimodal_support", "Onco Breast Cancer Multimodal Support"),
    (52, "30_disease_use_cases", "52_onco_prostate_mri_disease_reasoning", "Onco Prostate MRI Disease Reasoning"),
    (53, "30_disease_use_cases", "53_onco_whole_body_disease_burden", "Onco Whole Body Disease Burden"),
    (54, "30_disease_use_cases", "54_msk_trauma_multisite_fracture_detection", "MSK Trauma Multisite Fracture Detection"),
    (55, "30_disease_use_cases", "55_msk_osteoarthritis_joint_assessment", "MSK Osteoarthritis Joint Assessment"),
    (56, "30_disease_use_cases", "56_msk_spine_degeneration_and_stenosis", "MSK Spine Degeneration and Stenosis"),
    (57, "30_disease_use_cases", "57_msk_sports_injury_assessment", "MSK Sports Injury Assessment"),
    (58, "30_disease_use_cases", "58_chest_infection_multimodal_support", "Chest Infection Multimodal Support"),
    (59, "30_disease_use_cases", "59_chest_ild_pattern_support", "Chest ILD Pattern Support"),
    (60, "30_disease_use_cases", "60_chest_pe_ct_and_xray_context", "Chest PE CT and X-ray Context"),
    (61, "30_disease_use_cases", "61_chest_covid_like_patterns", "Chest COVID-like Patterns"),
    
    # 40_xai_and_dossiers (62-69)
    (62, "40_xai_and_dossiers", "62_xai_methods_and_apis_overview", "XAI Methods and APIs Overview"),
    (63, "40_xai_and_dossiers", "63_xai_visual_saliency_and_gradcam", "XAI Visual Saliency and Grad-CAM"),
    (64, "40_xai_and_dossiers", "64_xai_region_level_explanations", "XAI Region Level Explanations"),
    (65, "40_xai_and_dossiers", "65_xai_disease_level_linking", "XAI Disease Level Linking"),
    (66, "40_xai_and_dossiers", "66_evidence_dossiers_single_study", "Evidence Dossiers Single Study"),
    (67, "40_xai_and_dossiers", "67_evidence_dossiers_longitudinal_case", "Evidence Dossiers Longitudinal Case"),
    (68, "40_xai_and_dossiers", "68_medgemma_narrative_generation_control", "MedGemma Narrative Generation Control"),
    (69, "40_xai_and_dossiers", "69_xai_quality_checks_and_sanity_tests", "XAI Quality Checks and Sanity Tests"),
    
    # 50_evaluation_and_benchmarks (70-77)
    (70, "50_evaluation_and_benchmarks", "70_evaluation_metrics_and_api_overview", "Evaluation Metrics and API Overview"),
    (71, "50_evaluation_and_benchmarks", "71_evaluation_segmentation_benchmarks", "Evaluation Segmentation Benchmarks"),
    (72, "50_evaluation_and_benchmarks", "72_evaluation_classification_benchmarks", "Evaluation Classification Benchmarks"),
    (73, "50_evaluation_and_benchmarks", "73_evaluation_detection_benchmarks", "Evaluation Detection Benchmarks"),
    (74, "50_evaluation_and_benchmarks", "74_evaluation_trajectory_benchmarks", "Evaluation Trajectory Benchmarks"),
    (75, "50_evaluation_and_benchmarks", "75_evaluation_imaging_quality_benchmarks", "Evaluation Imaging Quality Benchmarks"),
    (76, "50_evaluation_and_benchmarks", "76_evaluation_cross_modality_comparisons", "Evaluation Cross Modality Comparisons"),
    (77, "50_evaluation_and_benchmarks", "77_evaluation_reporting_and_export", "Evaluation Reporting and Export"),
    
    # 60_fairness_and_bias (78-83)
    (78, "60_fairness_and_bias", "78_fairness_conceptual_overview", "Fairness Conceptual Overview"),
    (79, "60_fairness_and_bias", "79_fairness_demographic_stratification", "Fairness Demographic Stratification"),
    (80, "60_fairness_and_bias", "80_fairness_site_and_scanner_domain_shift", "Fairness Site and Scanner Domain Shift"),
    (81, "60_fairness_and_bias", "81_fairness_protocol_and_quality_stratification", "Fairness Protocol and Quality Stratification"),
    (82, "60_fairness_and_bias", "82_fairness_mitigation_experiments", "Fairness Mitigation Experiments"),
    (83, "60_fairness_and_bias", "83_governance_reports_and_audit_trails", "Governance Reports and Audit Trails"),
    
    # 70_ml_engineering_and_ablations (84-91)
    (84, "70_ml_engineering_and_ablations", "84_training_pipelines_overview", "Training Pipelines Overview"),
    (85, "70_ml_engineering_and_ablations", "85_training_hyperparameter_optimization", "Training Hyperparameter Optimization"),
    (86, "70_ml_engineering_and_ablations", "86_ablation_architecture_variants", "Ablation Architecture Variants"),
    (87, "70_ml_engineering_and_ablations", "87_ablation_data_augmentation", "Ablation Data Augmentation"),
    (88, "70_ml_engineering_and_ablations", "88_transfer_learning_and_pretraining", "Transfer Learning and Pretraining"),
    (89, "70_ml_engineering_and_ablations", "89_experiment_tracking_and_model_versioning", "Experiment Tracking and Model Versioning"),
    (90, "70_ml_engineering_and_ablations", "90_robustness_testing_and_stress_tests", "Robustness Testing and Stress Tests"),
    (91, "70_ml_engineering_and_ablations", "91_model_packaging_and_export", "Model Packaging and Export"),
    
    # 80_integration_and_workflows (92-98)
    (92, "80_integration_and_workflows", "92_integration_pacs_and_dicom_workflows", "Integration PACS and DICOM Workflows"),
    (93, "80_integration_and_workflows", "93_integration_batch_inference_workflows", "Integration Batch Inference Workflows"),
    (94, "80_integration_and_workflows", "94_integration_async_queue_based_pipelines", "Integration Async Queue Based Pipelines"),
    (95, "80_integration_and_workflows", "95_integration_clinical_workflow_simulation", "Integration Clinical Workflow Simulation"),
    (96, "80_integration_and_workflows", "96_integration_monitoring_and_logging", "Integration Monitoring and Logging"),
    (97, "80_integration_and_workflows", "97_integration_continuous_evaluation", "Integration Continuous Evaluation"),
    (98, "80_integration_and_workflows", "98_integration_api_clients_and_sdk_usage", "Integration API Clients and SDK Usage"),
    
    # 90_misc (99-100)
    (99, "90_misc", "99_meta_reproducibility_and_environment", "Meta Reproducibility and Environment"),
    (100, "90_misc", "100_meta_notebook_template_generator", "Meta Notebook Template Generator"),
]

def create_notebook(nb_id, title):
    """Generate a Jupyter notebook structure."""
    disclaimer = "**Disclaimer:** This is not a medical device. Outputs are for research and development only."
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Notebook {nb_id:02d} - {title}\n",
                    "\n",
                    f"This notebook covers: {title}\n",
                    "\n",
                    "**Runtime:** Variable  \n",
                    "**Dependencies:** rhenium, numpy, matplotlib\n",
                    "\n",
                    "---\n",
                    "\n",
                    f"{disclaimer}"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "sys.path.insert(0, '../..')\n",
                    "\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "from pathlib import Path\n",
                    "from datetime import datetime\n",
                    "\n",
                    "try:\n",
                    "    from rhenium.core import config\n",
                    "    print(f\"Rhenium OS loaded at {datetime.now()}\")\n",
                    "except ImportError as e:\n",
                    "    print(f\"Note: {e}\")\n",
                    "\n",
                    "np.random.seed(42)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Configuration"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"NOTEBOOK_ID = {nb_id}\n",
                    f"NOTEBOOK_TITLE = \"{title}\"\n",
                    "PROJECT_ROOT = Path('../..').resolve()\n",
                    "print(f\"Notebook {NOTEBOOK_ID}: {NOTEBOOK_TITLE}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Demonstration"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Placeholder for notebook-specific implementation\n",
                    f"print(\"Notebook {nb_id}: {title}\")\n",
                    "print(\"Implementation pending...\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Next Steps\n",
                    "\n",
                    "See [Index Notebook](../00_index/01_index_overview.ipynb) for navigation.\n",
                    "\n",
                    "---\n",
                    "\n",
                    "**Copyright (c) 2025 Skolyn LLC. All rights reserved.**"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return notebook

def main():
    base_dir = Path("notebooks")
    created = 0
    
    for nb_id, folder, filename, title in NOTEBOOKS:
        nb_path = base_dir / folder / f"{filename}.ipynb"
        nb_path.parent.mkdir(parents=True, exist_ok=True)
        
        notebook = create_notebook(nb_id, title)
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        created += 1
        print(f"Created: {nb_path}")
    
    print(f"\nTotal: {created} notebooks")

if __name__ == "__main__":
    main()
