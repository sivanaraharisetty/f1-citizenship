#!/usr/bin/env python3
"""
Focused Monitor for Production Sampling Process
Real-time monitoring with detailed progress tracking
"""

import time
import psutil
import os
from datetime import datetime
from pathlib import Path
import subprocess

def get_process_info(pid):
    """Get detailed process information"""
    try:
        process = psutil.Process(pid)
        return {
            'pid': pid,
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'status': process.status(),
            'create_time': datetime.fromtimestamp(process.create_time()),
            'num_threads': process.num_threads(),
            'connections': len(process.connections())
        }
    except psutil.NoSuchProcess:
        return None

def monitor_production_sampling():
    """Monitor the production sampling process in real-time"""
    print(" PRODUCTION SAMPLING MONITOR")
    print("=" * 60)
    
    # Find the production sampling process
    production_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'production_sampling.py' in ' '.join(proc.info['cmdline']):
                production_pid = proc.info['pid']
                break
        except:
            continue
    
    if not production_pid:
        print(" Production sampling process not found")
        return
    
    print(f" Found production sampling process: PID {production_pid}")
    print()
    
    # Monitor loop
    start_time = datetime.now()
    last_log_size = 0
    last_batch_count = 0
    
    try:
        while True:
            # Get process info
            proc_info = get_process_info(production_pid)
            if not proc_info:
                print(" Process no longer running")
                break
            
            # Calculate runtime
            runtime = datetime.now() - proc_info['create_time']
            
            # Check log file for new activity
            log_file = Path("sampled_data/sampling_log.txt")
            current_log_size = log_file.stat().st_size if log_file.exists() else 0
            log_activity = " Active" if current_log_size > last_log_size else "⏸  Quiet"
            last_log_size = current_log_size
            
            # Check batch files
            batch_dir = Path("sampled_data/batches")
            current_batch_count = len(list(batch_dir.glob("*.parquet"))) if batch_dir.exists() else 0
            batch_activity = " Saving" if current_batch_count > last_batch_count else "⏳ Processing"
            last_batch_count = current_batch_count
            
            # Display status
            print(f"\r {datetime.now().strftime('%H:%M:%S')} | "
                  f"⏱  {runtime} | "
                  f" CPU: {proc_info['cpu_percent']:.1f}% | "
                  f"🧠 RAM: {proc_info['memory_mb']:.1f}MB | "
                  f"🧵 Threads: {proc_info['num_threads']} | "
                  f" Connections: {proc_info['connections']} | "
                  f"{log_activity} | "
                  f" Batches: {current_batch_count} | "
                  f"{batch_activity}", end="", flush=True)
            
            # Check for completion
            if proc_info['status'] == psutil.STATUS_ZOMBIE:
                print("\n Process completed")
                break
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n Monitoring stopped by user")
    
    # Final status
    print(f"\n Final Status:")
    print(f"  Runtime: {datetime.now() - start_time}")
    print(f"  Final batches: {current_batch_count}")
    
    # Check output files
    output_files = list(Path("sampled_data").glob("*.parquet"))
    if output_files:
        print(f"  Output files: {len(output_files)}")
        for file in output_files:
            size_mb = file.stat().st_size / 1024 / 1024
            print(f"    {file.name}: {size_mb:.1f}MB")

if __name__ == "__main__":
    monitor_production_sampling()
