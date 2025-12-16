
import shutil
from pathlib import Path
from rhenium.models import RheniumCoreModel, RheniumCoreModelConfig, TaskType
from rhenium.testing.synthetic import SyntheticDataGenerator
from rhenium.core.clinical import AuditLogger, DicomValidator

def verify_clinical_mode():
    print("Verifying Clinical Mode...")
    
    # 1. Clean logs
    log_dir = Path("logs/audit")
    if log_dir.exists():
        shutil.rmtree(log_dir)
        
    # 2. Setup Model in Clinical Mode
    config = RheniumCoreModelConfig(
        device="cpu",
        clinical_mode=True,  # STRICT VALIDATION
        seed=42
    )
    model = RheniumCoreModel(config)
    model.initialize()
    
    # 3. Create Compliant Volume
    gen = SyntheticDataGenerator(seed=42)
    vol = gen.generate_volume(shape=(32, 32, 32), modality="MRI")
    
    # 4. Inject Missing Metadata (to trigger failure)
    # Synthetic generator creates some metadata, but let's ensure it fails first
    vol.metadata.patient_id = "" # Required tag missing
    
    print("Attempting run with invalid metadata (expecting failure)...")
    try:
        model.run(vol, TaskType.SEGMENTATION)
        print("ERROR: Should have failed validation!")
    except ValueError as e:
        print(f"SUCCESS: Caught expected validation error: {e}")
        
    # 5. Fix Metadata and Retry
    print("Fixing metadata and retrying...")
    vol.metadata.patient_id = "PID12345"
    vol.metadata.patient_name = "Test^Patient"
    vol.metadata.study_uid = "1.2.3.4.5"
    vol.metadata.series_uid = "1.2.3.4.6"
    vol.metadata.sop_uid = "1.2.3.4.7"
    
    # NOTE: DicomValidator checks dict representation of metadata. 
    # ImageVolume.metadata is a class. My core.py passes `volume.metadata.__dict__`.
    # Let's verify `Validation` passes now.
    
    try:
        result = model.run(vol, TaskType.SEGMENTATION)
        print("SUCCESS: Clinical run completed with valid data.")
    except Exception as e:
        print(f"ERROR: Run failed with valid data: {e}")
        
    # 6. Verify Audit Log
    logger = AuditLogger()
    if logger.current_log_file.exists():
        with open(logger.current_log_file, "r") as f:
            lines = f.readlines()
            print(f"Audit log contains {len(lines)} entries.")
            if len(lines) >= 2: # 1 rejected, 1 success (start)
                print("SUCCESS: Audit logging active.")
            else:
                print("WARNING: Audit log might be incomplete.")
    else:
        print("ERROR: Audit log file not found!")

if __name__ == "__main__":
    verify_clinical_mode()
