
import os
import subprocess

CACHE_FILE = "final_data.csv"  # Change to your final data file name

if os.path.exists(CACHE_FILE):
    print(f"Pipeline data already saved")
else:
    print("No file already saved. Running pipeline.")
    subprocess.run(["python", "data_processing_clean.py"], check=True)

print("Running model pipeline...")
subprocess.run(["python", "model_pipeline_position_features.py"], check=True)

print("Pipeline completed successfully")
