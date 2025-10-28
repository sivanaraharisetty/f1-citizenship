"""
Simple script to check and display live pipeline status
Run this to see current status of your pipeline
"""
import time
import json
import os
import psutil
from datetime import datetime, timedelta
from pathlib import Path

def check_pipeline_status():
    """Check and display current pipeline status"""
    
    print(" Checking pipeline status...")
    print("=" * 60)
    
    # Check if pipeline is running
    pipeline_running = False
    process_info = {}
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info', 'create_time']):
        try:
            if 'python' in proc.info['name'] and 'run_full_analysis' in ' '.join(proc.cmdline()):
                pipeline_running = True
                process_info = {
                    'pid': proc.info['pid'],
                    'cpu_percent': proc.info['cpu_percent'],
                    'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                    'start_time': datetime.fromtimestamp(proc.info['create_time']).strftime('%H:%M:%S'),
                    'elapsed': str(datetime.now() - datetime.fromtimestamp(proc.info['create_time'])).split('.')[0]
                }
                break
        except:
            continue
    
    # Display process status
    if pipeline_running:
        print(f" PIPELINE STATUS: Running")
        print(f"   PID: {process_info['pid']}")
        print(f"   Started: {process_info['start_time']}")
        print(f"   Elapsed: {process_info['elapsed']}")
        print(f"   CPU: {process_info['cpu_percent']:.1f}%")
        print(f"   Memory: {process_info['memory_mb']:.1f} MB")
    else:
        print(" PIPELINE STATUS: Not running")
    
    print()
    
    # Check output files
    print(" OUTPUT FILES STATUS:")
    
    # Check sampled data
    sampled_dir = Path("sampled_data")
    if sampled_dir.exists():
        parquet_files = list(sampled_dir.glob("*.parquet"))
        if parquet_files:
            latest_file = max(parquet_files, key=os.path.getctime)
            size_mb = latest_file.stat().st_size / 1024 / 1024
            modified_time = datetime.fromtimestamp(latest_file.stat().st_mtime).strftime("%H:%M:%S")
            print(f"    Sampled Data: {latest_file.name} ({size_mb:.1f} MB) - {modified_time}")
        else:
            print("   ⏳ Sampled Data: No files yet")
    else:
        print("   ⏳ Sampled Data: Directory not found")
    
    # Check cleaned data
    cleaned_dir = Path("cleaned_data")
    if cleaned_dir.exists():
        parquet_files = list(cleaned_dir.glob("*.parquet"))
        if parquet_files:
            latest_file = max(parquet_files, key=os.path.getctime)
            size_mb = latest_file.stat().st_size / 1024 / 1024
            modified_time = datetime.fromtimestamp(latest_file.stat().st_mtime).strftime("%H:%M:%S")
            print(f"    Cleaned Data: {latest_file.name} ({size_mb:.1f} MB) - {modified_time}")
        else:
            print("   ⏳ Cleaned Data: No files yet")
    else:
        print("   ⏳ Cleaned Data: Directory not found")
    
    # Check classifier outputs
    classifier_dir = Path("classifier")
    if classifier_dir.exists():
        model_files = list(classifier_dir.glob("**/*.pkl")) + list(classifier_dir.glob("**/*.bin"))
        if model_files:
            latest_file = max(model_files, key=os.path.getctime)
            modified_time = datetime.fromtimestamp(latest_file.stat().st_mtime).strftime("%H:%M:%S")
            print(f"    Classifier: {len(model_files)} files - Latest: {latest_file.name} ({modified_time})")
        else:
            print("   ⏳ Classifier: No model files yet")
    else:
        print("   ⏳ Classifier: Directory not found")
    
    # Check visualizations
    viz_dir = Path("visualizations")
    if viz_dir.exists():
        viz_files = list(viz_dir.glob("**/*"))
        if viz_files:
            latest_file = max(viz_files, key=os.path.getctime)
            modified_time = datetime.fromtimestamp(latest_file.stat().st_mtime).strftime("%H:%M:%S")
            print(f"    Visualizations: {len(viz_files)} files - Latest: {latest_file.name} ({modified_time})")
        else:
            print("   ⏳ Visualizations: No files yet")
    else:
        print("   ⏳ Visualizations: Directory not found")
    
    print()
    
    # Pipeline steps status
    print(" PIPELINE STEPS STATUS:")
    steps = [
        ("Data Sampling", "sampled_data" in str(sampled_dir) and sampled_dir.exists() and list(sampled_dir.glob("*.parquet"))),
        ("Data Cleaning", "cleaned_data" in str(cleaned_dir) and cleaned_dir.exists() and list(cleaned_dir.glob("*.parquet"))),
        ("Descriptive Analysis", False),  # Will be updated when implemented
        ("Annotation", False),
        ("BERT Training", "classifier" in str(classifier_dir) and classifier_dir.exists() and list(classifier_dir.glob("**/*.pkl"))),
        ("Evaluation", False),
        ("Temporal Analysis", False),
        ("Visualization", "visualizations" in str(viz_dir) and viz_dir.exists() and list(viz_dir.glob("**/*")))
    ]
    
    for step, completed in steps:
        status_icon = "" if completed else "⏳"
        print(f"   {status_icon} {step}")
    
    print()
    
    # Estimated completion
    if pipeline_running:
        elapsed_minutes = int(process_info['elapsed'].split(':')[0]) * 60 + int(process_info['elapsed'].split(':')[1])
        estimated_total_minutes = 120  # 2 hours estimate
        remaining_minutes = max(0, estimated_total_minutes - elapsed_minutes)
        
        if remaining_minutes > 0:
            remaining_hours = remaining_minutes // 60
            remaining_mins = remaining_minutes % 60
            print(f" ESTIMATED COMPLETION: {remaining_hours}h {remaining_mins}m remaining")
        else:
            print(" ESTIMATED COMPLETION: Should be completing soon")
    else:
        print(" ESTIMATED COMPLETION: Pipeline not running")
    
    print("=" * 60)

def monitor_continuously():
    """Monitor pipeline status continuously"""
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            check_pipeline_status()
            print(f"\n Next update in 30 seconds... (Press Ctrl+C to stop)")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n Monitoring stopped")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        monitor_continuously()
    else:
        check_pipeline_status()
