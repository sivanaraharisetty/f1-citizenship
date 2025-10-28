#!/usr/bin/env python3
"""
Production Sampling Monitor
A macOS-compatible monitoring script that provides real-time status updates
"""

import os
import sys
import time
import subprocess
import glob
from datetime import datetime
from pathlib import Path

def clear_screen():
    """Clear the terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def get_process_info():
    """Get information about running production sampling processes"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        production_processes = [line for line in lines if 'production_sampling' in line and 'grep' not in line]
        return production_processes
    except Exception as e:
        return [f"Error getting process info: {e}"]

def get_latest_log_entries(n=5):
    """Get the latest n entries from the sampling log"""
    log_file = Path("sampled_data/sampling_log.txt")
    if not log_file.exists():
        return ["No log file found"]
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return lines[-n:] if lines else ["Log file is empty"]
    except Exception as e:
        return [f"Error reading log: {e}"]

def get_batch_files():
    """Get information about batch files"""
    batch_dir = Path("sampled_data/batches")
    if not batch_dir.exists():
        return ["No batches directory found"]
    
    try:
        files = list(batch_dir.glob("*"))
        if not files:
            return ["No batch files yet"]
        
        # Get file info
        file_info = []
        for file_path in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            stat = file_path.stat()
            size = stat.st_size
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            file_info.append(f"{file_path.name:<30} {size:>10} bytes  {mtime}")
        return file_info
    except Exception as e:
        return [f"Error reading batch files: {e}"]

def get_output_files():
    """Get information about output parquet files"""
    try:
        parquet_files = glob.glob("sampled_data/*.parquet")
        if not parquet_files:
            return ["No parquet files found"]
        
        file_info = []
        for file_path in sorted(parquet_files, key=lambda x: os.path.getmtime(x), reverse=True)[:3]:
            stat = os.stat(file_path)
            size = stat.st_size
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            file_info.append(f"{os.path.basename(file_path):<30} {size:>10} bytes  {mtime}")
        return file_info
    except Exception as e:
        return [f"Error reading output files: {e}"]

def display_status():
    """Display the current status"""
    clear_screen()
    
    print("=" * 60)
    print(" PRODUCTION SAMPLING MONITOR")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Process information
    print("\n RUNNING PROCESSES:")
    processes = get_process_info()
    if processes:
        for process in processes:
            print(f"  {process}")
    else:
        print("  No production sampling processes found")
    
    # Latest log entries
    print("\n LATEST LOG ENTRIES:")
    log_entries = get_latest_log_entries()
    for entry in log_entries:
        print(f"  {entry.strip()}")
    
    # Batch files
    print("\n BATCH FILES:")
    batch_files = get_batch_files()
    for file_info in batch_files:
        print(f"  {file_info}")
    
    # Output files
    print("\n OUTPUT FILES:")
    output_files = get_output_files()
    for file_info in output_files:
        print(f"  {file_info}")
    
    print("\n" + "=" * 60)
    print("Press Ctrl+C to exit")

def main():
    """Main monitoring loop"""
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Run once and exit
        display_status()
        return
    
    try:
        print("Starting production sampling monitor...")
        print("Press Ctrl+C to exit")
        time.sleep(2)
        
        while True:
            display_status()
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\n Monitor stopped by user")
    except Exception as e:
        print(f"\n Error in monitor: {e}")

if __name__ == "__main__":
    main()
