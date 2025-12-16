"""Rhenium OS Unified Core Model.

This module provides the central entrypoint for the Rhenium OS platform,
integrating perception (segmentation, classification, detection), reconstruction
(MRI, CT baselines), generative (super-resolution, denoising, GANs), and XAI
(evidence dossier generation) subsystems into a single configurable model.

IMPORTANT: This is a research and development system. It is NOT intended for
clinical use and makes NO claims of clinical performance or regulatory compliance.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rhenium.core.registry import registry
from rhenium.core.reproducibility import set_seed, get_deterministic_context
from rhenium.core.clinical import AuditLogger, DicomValidator, ClinicalSafety
from rhenium.data.volume import ImageVolume, Modality
from rhenium.generative.disclosure import GenerationMetadata
from rhenium.xai import (
    EvidenceDossier,
    Finding,
    QuantitativeEvidence,
    NarrativeEvidence,
    VisualEvidence,
)


class TaskType(str, Enum):
    """Supported task types for the core model."""

    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    RECONSTRUCTION = "reconstruction"
    SUPER_RESOLUTION = "super_resolution"
    DENOISE = "denoise"
    FULL_PIPELINE = "full_pipeline"


@dataclass
class RheniumCoreModelConfig:
    """Configuration for RheniumCoreModel.

    Attributes:
        device: Compute device ("cpu" or "cuda")
        seed: Random seed for reproducibility
        deterministic: Enable deterministic operations
        perception_enabled: Enable perception subsystem
        reconstruction_enabled: Enable reconstruction subsystem
        generative_enabled: Enable generative subsystem
        xai_enabled: Enable XAI/evidence dossier generation
        segmentation_model: Name of segmentation model to use
        segmentation_features: Feature sizes for segmentation model
        reconstruction_model: Name of reconstruction model
        generator_model: Name of generator model for SR/denoise
        dtype: Data type for computation
    """

    device: str = "cpu"
    seed: int = 42
    deterministic: bool = True
    perception_enabled: bool = True
    reconstruction_enabled: bool = True
    generative_enabled: bool = True
    reconstruction_enabled: bool = True
    generative_enabled: bool = True
    xai_enabled: bool = True
    clinical_mode: bool = False  # Enable strict clinical validation logic

    # Model configurations (small defaults for testing)
    segmentation_model: str = "unet3d"
    segmentation_features: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    segmentation_classes: int = 2

    reconstruction_model: str = "mri_zerofilled"

    generator_model: str = "srgan"
    generator_features: int = 32
    generator_rrdb_blocks: int = 4
    upscale_factor: int = 2

    dtype: str = "float32"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.device not in ("cpu", "cuda"):
            if not self.device.startswith("cuda:"):
                raise ValueError(f"Invalid device: {self.device}")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"Invalid dtype: {self.dtype}")


@dataclass
class CoreModelOutput:
    """Structured output from RheniumCoreModel.

    Attributes:
        task: Task that was performed
        output: Primary output (segmentation mask, reconstructed image, etc.)
        evidence_dossier: XAI evidence dossier if enabled
        generation_metadata: Metadata for generative outputs
        provenance: Provenance information for audit trail
        metrics: Computed metrics
        raw_outputs: Additional raw outputs from subsystems
    """

    task: TaskType
    output: np.ndarray | torch.Tensor
    evidence_dossier: dict[str, Any] | None = None
    generation_metadata: dict[str, Any] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    raw_outputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        output_data = self.output
        if isinstance(output_data, torch.Tensor):
            output_data = output_data.cpu().numpy()

        return {
            "task": self.task.value,
            "output_shape": list(output_data.shape),
            "output_dtype": str(output_data.dtype),
            "evidence_dossier": self.evidence_dossier,
            "generation_metadata": self.generation_metadata,
            "provenance": self.provenance,
            "metrics": self.metrics,
        }


class RheniumCoreModel:
    """Unified core model integrating all Rhenium OS subsystems.

    This class provides a single entrypoint for medical imaging AI tasks,
    dispatching to the appropriate subsystem based on the requested task.

    Example:
        >>> config = RheniumCoreModelConfig(device="cpu", seed=42)
        >>> model = RheniumCoreModel(config)
        >>> model.initialize()
        >>> result = model.run(volume, task=TaskType.SEGMENTATION)
        >>> print(result.output.shape)

    Note:
        This is for research and development only. Not for clinical use.
    """

    VERSION = "1.0.0"

    def __init__(self, config: RheniumCoreModelConfig) -> None:
        """Initialize the core model.

        Args:
            config: Model configuration
        """
        self.config = config
        self._initialized = False
        self._models: dict[str, nn.Module] = {}
        self._run_counter = 0
        self._audit_logger = AuditLogger() if config.clinical_mode else None

    def initialize(self) -> None:
        """Initialize all enabled subsystems.

        Must be called before run().
        """
        # Set reproducibility
        if self.config.deterministic:
            set_seed(self.config.seed)

        # Check CUDA availability
        if self.config.device == "cuda" and not torch.cuda.is_available():
            self.config.device = "cpu"

        # Load models based on enabled subsystems
        if self.config.perception_enabled:
            self._load_perception_models()

        if self.config.reconstruction_enabled:
            self._load_reconstruction_models()

        if self.config.generative_enabled:
            self._load_generative_models()

        self._initialized = True

    def _load_perception_models(self) -> None:
        """Load perception subsystem models."""
        try:
            model_cls = registry.get("model", self.config.segmentation_model)
            model = model_cls(
                in_channels=1,
                out_channels=self.config.segmentation_classes,
                features=self.config.segmentation_features,
            )
            model = model.to(self.config.device)
            model.eval()
            self._models["segmentation"] = model
        except KeyError:
            # Model not registered, use minimal fallback
            pass

    def _load_reconstruction_models(self) -> None:
        """Load reconstruction subsystem models."""
        try:
            model_cls = registry.get("reconstructor", self.config.reconstruction_model)
            model = model_cls()
            if hasattr(model, "to"):
                model = model.to(self.config.device)
            self._models["reconstruction"] = model
        except KeyError:
            pass

    def _load_generative_models(self) -> None:
        """Load generative subsystem models."""
        try:
            model_cls = registry.get("generator", self.config.generator_model)
            model = model_cls(
                in_ch=1,
                out_ch=1,
                features=self.config.generator_features,
                n_rrdb=self.config.generator_rrdb_blocks,
                upscale=self.config.upscale_factor,
            )
            model = model.to(self.config.device)
            model.eval()
            self._models["generator"] = model
        except KeyError:
            pass

    def run(
        self,
        volume: ImageVolume,
        task: TaskType | str,
        **kwargs: Any,
    ) -> CoreModelOutput:
        """Execute the specified task on the input volume.

        Args:
            volume: Input medical image volume
            task: Task to perform
            **kwargs: Additional task-specific arguments

        Returns:
            Structured output with results and metadata

        Raises:
            RuntimeError: If model not initialized or task not supported
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        if isinstance(task, str):
            task = TaskType(task)

        self._run_counter += 1

        # Run Clinical Safety Checks
        if self.config.clinical_mode:
            if not ClinicalSafety.verify_memory_headroom():
                if self._audit_logger:
                    self._audit_logger.log("SAFETY_CHECK", "system", "host", "FAILURE", {"error": "Insufficient Memory"})
                raise RuntimeError("Insufficient memory for clinical safety.")
            
            # Validate Input Metadata
            is_valid, errors = DicomValidator.validate_metadata(volume.metadata.__dict__)
            if not is_valid:
                if self._audit_logger:
                    self._audit_logger.log(f"TASK_REQUEST_{task.name}", "user", volume.metadata.study_uid or "unknown", "REJECTED", {"errors": errors})
                raise ValueError(f"Clinical validation failed: {errors}")

            # Prepare Audit Log
            if self._audit_logger:
                self._audit_logger.log(f"TASK_START_{task.name}", "user", volume.metadata.study_uid or "unknown", "STARTED", {"run_id": self._run_counter})

        # Set deterministic context for this run
        if self.config.deterministic:
            set_seed(self.config.seed + self._run_counter - 1)

        # Dispatch to appropriate handler
        dispatch = {
            TaskType.SEGMENTATION: self._run_segmentation,
            TaskType.CLASSIFICATION: self._run_classification,
            TaskType.DETECTION: self._run_detection,
            TaskType.RECONSTRUCTION: self._run_reconstruction,
            TaskType.SUPER_RESOLUTION: self._run_super_resolution,
            TaskType.DENOISE: self._run_denoise,
            TaskType.FULL_PIPELINE: self._run_full_pipeline,
        }

        handler = dispatch.get(task)
        if handler is None:
            raise ValueError(f"Unsupported task: {task}")

        return handler(volume, **kwargs)

    def _run_segmentation(self, volume: ImageVolume, **kwargs: Any) -> CoreModelOutput:
        """Run segmentation task."""
        model = self._models.get("segmentation")

        # Convert to tensor
        tensor = volume.to_tensor(device=self.config.device)

        if model is not None:
            with torch.no_grad():
                logits = model(tensor)
                mask = logits.argmax(dim=1).squeeze(0)
        else:
            # Fallback: simple thresholding
            mask = (tensor.squeeze() > 0.5).long()

        mask_np = mask.cpu().numpy()

        # Generate evidence dossier if enabled
        evidence_dossier = None
        if self.config.xai_enabled:
            evidence_dossier = self._generate_evidence_dossier(
                volume, mask_np, task="segmentation", **kwargs
            )

        return CoreModelOutput(
            task=TaskType.SEGMENTATION,
            output=mask_np,
            evidence_dossier=evidence_dossier,
            provenance=self._create_provenance(volume, "segmentation"),
            metrics={"volume_voxels": int(mask_np.sum())},
        )

    def _run_classification(self, volume: ImageVolume, **kwargs: Any) -> CoreModelOutput:
        """Run classification task."""
        # Simple stub - would integrate classification model
        tensor = volume.to_tensor(device=self.config.device)

        # Placeholder classification based on intensity
        mean_intensity = tensor.mean().item()
        class_idx = 0 if mean_intensity < 0.5 else 1
        confidence = abs(mean_intensity - 0.5) * 2

        output = np.array([class_idx])

        evidence_dossier = None
        if self.config.xai_enabled:
            evidence_dossier = self._generate_evidence_dossier(
                volume, output, task="classification", confidence=confidence, **kwargs
            )

        return CoreModelOutput(
            task=TaskType.CLASSIFICATION,
            output=output,
            evidence_dossier=evidence_dossier,
            provenance=self._create_provenance(volume, "classification"),
            metrics={"confidence": confidence, "predicted_class": int(class_idx)},
        )

    def _run_detection(self, volume: ImageVolume, **kwargs: Any) -> CoreModelOutput:
        """Run detection task."""
        # Simple stub - detect regions above threshold
        threshold = kwargs.get("threshold", 0.7)
        array = volume.array

        # Find connected components above threshold (simplified)
        detections = []
        mask = array > threshold
        if mask.any():
            coords = np.argwhere(mask)
            if len(coords) > 0:
                center = coords.mean(axis=0).tolist()
                detections.append({
                    "center": center,
                    "confidence": float(array[mask].mean()),
                    "bbox": [
                        int(coords[:, i].min()) for i in range(3)
                    ] + [
                        int(coords[:, i].max()) for i in range(3)
                    ],
                })

        output = np.array(detections) if detections else np.array([])

        evidence_dossier = None
        if self.config.xai_enabled:
            evidence_dossier = self._generate_evidence_dossier(
                volume, output, task="detection", detections=detections, **kwargs
            )

        return CoreModelOutput(
            task=TaskType.DETECTION,
            output=output,
            evidence_dossier=evidence_dossier,
            provenance=self._create_provenance(volume, "detection"),
            metrics={"num_detections": len(detections)},
            raw_outputs={"detections": detections},
        )

    def _run_reconstruction(self, volume: ImageVolume, **kwargs: Any) -> CoreModelOutput:
        """Run reconstruction task."""
        model = self._models.get("reconstruction")

        # For reconstruction, expect k-space or sinogram input
        kspace = kwargs.get("kspace")
        mask = kwargs.get("mask")

        if kspace is not None:
            kspace_tensor = torch.from_numpy(kspace).to(self.config.device)
            if model is not None and mask is not None:
                mask_tensor = torch.from_numpy(mask).to(self.config.device)
                with torch.no_grad():
                    reconstructed = model(kspace_tensor, mask=mask_tensor)
            else:
                # Zero-filled reconstruction
                from rhenium.reconstruction.base import ifft2c
                reconstructed = ifft2c(kspace_tensor).abs()

            output = reconstructed.cpu().numpy()
        else:
            # Pass-through if no k-space provided
            output = volume.array

        return CoreModelOutput(
            task=TaskType.RECONSTRUCTION,
            output=output,
            provenance=self._create_provenance(volume, "reconstruction"),
            metrics={"output_range": [float(output.min()), float(output.max())]},
        )

    def _run_super_resolution(self, volume: ImageVolume, **kwargs: Any) -> CoreModelOutput:
        """Run super-resolution task."""
        model = self._models.get("generator")

        # Handle 2D slices for SR
        array = volume.array
        if array.ndim == 3:
            # Process middle slice as 2D
            mid_slice = array[array.shape[0] // 2]
            tensor = torch.from_numpy(mid_slice).float().unsqueeze(0).unsqueeze(0)
        else:
            tensor = torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)

        tensor = tensor.to(self.config.device)

        if model is not None:
            with torch.no_grad():
                sr_output = model(tensor)
        else:
            # Bilinear upscale fallback
            sr_output = torch.nn.functional.interpolate(
                tensor, scale_factor=self.config.upscale_factor, mode="bilinear"
            )

        output = sr_output.squeeze().cpu().numpy()

        # Generate disclosure metadata for generated content
        generation_metadata = self._create_generation_metadata(
            volume, "super_resolution"
        )

        evidence_dossier = None
        if self.config.xai_enabled:
            evidence_dossier = self._generate_evidence_dossier(
                volume, output, task="super_resolution", **kwargs
            )

        return CoreModelOutput(
            task=TaskType.SUPER_RESOLUTION,
            output=output,
            evidence_dossier=evidence_dossier,
            generation_metadata=generation_metadata,
            provenance=self._create_provenance(volume, "super_resolution"),
            metrics={"upscale_factor": self.config.upscale_factor},
        )

    def _run_denoise(self, volume: ImageVolume, **kwargs: Any) -> CoreModelOutput:
        """Run denoising task."""
        # Simple denoising using Gaussian filter
        from scipy.ndimage import gaussian_filter

        sigma = kwargs.get("sigma", 1.0)
        denoised = gaussian_filter(volume.array, sigma=sigma)

        generation_metadata = self._create_generation_metadata(volume, "denoise")

        evidence_dossier = None
        if self.config.xai_enabled:
            evidence_dossier = self._generate_evidence_dossier(
                volume, denoised, task="denoise", **kwargs
            )

        return CoreModelOutput(
            task=TaskType.DENOISE,
            output=denoised,
            evidence_dossier=evidence_dossier,
            generation_metadata=generation_metadata,
            provenance=self._create_provenance(volume, "denoise"),
            metrics={"sigma": sigma},
        )

    def _run_full_pipeline(self, volume: ImageVolume, **kwargs: Any) -> CoreModelOutput:
        """Run full pipeline: preprocess -> segment -> generate dossier."""
        # Step 1: Normalize
        normalized = volume.normalize(method="minmax")

        # Step 2: Segment
        seg_result = self._run_segmentation(normalized, **kwargs)

        # Step 3: Compute measurements
        mask = seg_result.output
        spacing_product = np.prod(volume.spacing)
        volume_mm3 = float(mask.sum() * spacing_product)

        # Combine outputs
        return CoreModelOutput(
            task=TaskType.FULL_PIPELINE,
            output=mask,
            evidence_dossier=seg_result.evidence_dossier,
            provenance=self._create_provenance(volume, "full_pipeline"),
            metrics={
                "segmentation_volume_voxels": int(mask.sum()),
                "segmentation_volume_mm3": volume_mm3,
            },
            raw_outputs={"segmentation_result": seg_result.to_dict()},
        )

    def _generate_evidence_dossier(
        self,
        volume: ImageVolume,
        output: np.ndarray,
        task: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate XAI evidence dossier for the output."""
        dossier_id = str(uuid.uuid4())[:8]
        study_uid = kwargs.get("study_uid", volume.metadata.study_uid or "unknown")
        series_uid = kwargs.get("series_uid", volume.metadata.series_uid or "unknown")

        # Create finding based on task
        if task == "segmentation":
            volume_value = float(np.sum(output) * np.prod(volume.spacing))
            finding = Finding(
                finding_id=f"seg_{dossier_id}",
                finding_type="segmentation_mask",
                description=f"Automated {task} result",
                confidence=kwargs.get("confidence", 0.85),
                quantitative_evidence=[
                    QuantitativeEvidence(
                        evidence_id=f"vol_{dossier_id}",
                        evidence_type="volume",
                        value=volume_value,
                        unit="mm3",
                    )
                ],
                narrative_evidence=[
                    NarrativeEvidence(
                        evidence_id=f"narr_{dossier_id}",
                        explanation=f"Segmentation produced mask with {int(output.sum())} positive voxels",
                        limitations=["Research use only", "Not validated for clinical decisions"],
                    )
                ],
            )
        elif task == "classification":
            finding = Finding(
                finding_id=f"cls_{dossier_id}",
                finding_type="classification_result",
                description="Automated classification result",
                confidence=kwargs.get("confidence", 0.5),
                narrative_evidence=[
                    NarrativeEvidence(
                        evidence_id=f"narr_{dossier_id}",
                        explanation=f"Classification predicted class {int(output[0]) if len(output) > 0 else 'N/A'}",
                        limitations=["Research use only"],
                    )
                ],
            )
        elif task == "detection":
            detections = kwargs.get("detections", [])
            finding = Finding(
                finding_id=f"det_{dossier_id}",
                finding_type="detection_result",
                description=f"Detected {len(detections)} region(s)",
                confidence=kwargs.get("confidence", 0.75),
                narrative_evidence=[
                    NarrativeEvidence(
                        evidence_id=f"narr_{dossier_id}",
                        explanation=f"Detection found {len(detections)} candidate regions",
                        limitations=["Research use only", "Requires expert review"],
                    )
                ],
            )
        else:
            finding = Finding(
                finding_id=f"gen_{dossier_id}",
                finding_type=f"{task}_result",
                description=f"Automated {task} result",
                confidence=kwargs.get("confidence", 0.9),
                narrative_evidence=[
                    NarrativeEvidence(
                        evidence_id=f"narr_{dossier_id}",
                        explanation=f"Task {task} completed successfully",
                        limitations=["Research use only"],
                    )
                ],
            )

        dossier = EvidenceDossier(
            dossier_id=dossier_id,
            finding=finding,
            study_uid=study_uid,
            series_uid=series_uid,
            pipeline_name="RheniumCoreModel",
            pipeline_version=self.VERSION,
        )

        return dossier.to_dict()

    def _create_generation_metadata(
        self,
        volume: ImageVolume,
        task: str,
    ) -> dict[str, Any]:
        """Create generation metadata for generative outputs."""
        input_hash = hashlib.sha256(volume.array.tobytes()[:1024]).hexdigest()[:16]

        metadata = GenerationMetadata(
            generator_name="RheniumCoreModel",
            generator_version=self.VERSION,
            input_hash=input_hash,
            parameters={
                "task": task,
                "device": self.config.device,
                "seed": self.config.seed,
            },
            disclosure="This image was generated by AI and is for research purposes only",
        )

        return metadata.to_dict()

    def _create_provenance(
        self,
        volume: ImageVolume,
        task: str,
    ) -> dict[str, Any]:
        """Create provenance metadata for audit trail."""
        return {
            "model_name": "RheniumCoreModel",
            "model_version": self.VERSION,
            "task": task,
            "device": self.config.device,
            "seed": self.config.seed,
            "deterministic": self.config.deterministic,
            "input_shape": list(volume.shape),
            "input_modality": volume.modality.value,
            "input_spacing": list(volume.spacing),
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": self._run_counter,
        }

    def reset(self) -> None:
        """Reset model state for fresh runs."""
        self._run_counter = 0
        if self.config.deterministic:
            set_seed(self.config.seed)

    def shutdown(self) -> None:
        """Clean up resources."""
        self._models.clear()
        self._initialized = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_initialized(self) -> bool:
        """Check if model is initialized."""
        return self._initialized

    @property
    def available_tasks(self) -> list[TaskType]:
        """Return list of available tasks based on loaded models."""
        tasks = []
        if "segmentation" in self._models or self.config.perception_enabled:
            tasks.extend([TaskType.SEGMENTATION, TaskType.CLASSIFICATION, TaskType.DETECTION])
        if "reconstruction" in self._models or self.config.reconstruction_enabled:
            tasks.append(TaskType.RECONSTRUCTION)
        if "generator" in self._models or self.config.generative_enabled:
            tasks.extend([TaskType.SUPER_RESOLUTION, TaskType.DENOISE])
        tasks.append(TaskType.FULL_PIPELINE)
        return tasks
