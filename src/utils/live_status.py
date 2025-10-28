"""
Live Status Monitor for Reddit Visa Discourse Analysis Pipeline
Generates human-readable status updates and saves to file
"""
import time
import json
import os
import psutil
from datetime import datetime, timedelta
from pathlib import Path

class LiveStatusMonitor:
    """Live status monitor that generates human-readable status updates"""
    
    def __init__(self, status_file="live_pipeline_status.txt"):
        self.status_file = status_file
        self.pipeline_start_time = None
        self.last_update = None
        
    def start_monitoring(self):
        """Start monitoring the pipeline"""
        self.pipeline_start_time = datetime.now()
        self.update_status(" Pipeline monitoring started")
        
    def update_status(self, message, step=None, progress=None, eta=None):
        """Update the status with a new message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Create status line
        if step and progress and eta:
            status_line = f"[{timestamp}] {message} | Step: {step} | Progress: {progress} | ETA: {eta}"
        elif step:
            status_line = f"[{timestamp}] {message} | Step: {step}"
        else:
            status_line = f"[{timestamp}] {message}"
        
        # Append to status file
        with open(self.status_file, 'a') as f:
            f.write(status_line + "\n")
        
        # Also print to console
        print(status_line)
        
        self.last_update = datetime.now()
    
    def get_system_info(self):
        """Get current system information"""
        try:
            # Get process info
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                if 'python' in proc.info['name'] and 'run_full_analysis' in ' '.join(proc.cmdline()):
                    return {
                        'pid': proc.info['pid'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                        'status': 'running'
                    }
        except:
            pass
        
        return {'status': 'not_found'}
    
    def check_output_files(self):
        """Check for output files and their status"""
        output_status = {}
        
        # Check sampled data
        sampled_dir = Path("sampled_data")
        if sampled_dir.exists():
            parquet_files = list(sampled_dir.glob("*.parquet"))
            if parquet_files:
                latest_file = max(parquet_files, key=os.path.getctime)
                size_mb = latest_file.stat().st_size / 1024 / 1024
                output_status['sampled_data'] = {
                    'file': latest_file.name,
                    'size_mb': round(size_mb, 2),
                    'modified': datetime.fromtimestamp(latest_file.stat().st_mtime).strftime("%H:%M:%S")
                }
            else:
                output_status['sampled_data'] = {'status': 'no_files'}
        else:
            output_status['sampled_data'] = {'status': 'directory_not_found'}
        
        # Check cleaned data
        cleaned_dir = Path("cleaned_data")
        if cleaned_dir.exists():
            parquet_files = list(cleaned_dir.glob("*.parquet"))
            if parquet_files:
                latest_file = max(parquet_files, key=os.path.getctime)
                size_mb = latest_file.stat().st_size / 1024 / 1024
                output_status['cleaned_data'] = {
                    'file': latest_file.name,
                    'size_mb': round(size_mb, 2),
                    'modified': datetime.fromtimestamp(latest_file.stat().st_mtime).strftime("%H:%M:%S")
                }
            else:
                output_status['cleaned_data'] = {'status': 'no_files'}
        else:
            output_status['cleaned_data'] = {'status': 'directory_not_found'}
        
        # Check classifier outputs
        classifier_dir = Path("classifier")
        if classifier_dir.exists():
            model_files = list(classifier_dir.glob("**/*.pkl")) + list(classifier_dir.glob("**/*.bin"))
            if model_files:
                output_status['classifier'] = {
                    'files': len(model_files),
                    'latest': max(model_files, key=os.path.getctime).name
                }
            else:
                output_status['classifier'] = {'status': 'no_files'}
        else:
            output_status['classifier'] = {'status': 'directory_not_found'}
        
        return output_status
    
    def generate_comprehensive_status(self):
        """Generate a comprehensive status report"""
        if not self.pipeline_start_time:
            return "Pipeline not started"
        
        # Get system info
        system_info = self.get_system_info()
        
        # Get output file status
        output_status = self.check_output_files()
        
        # Calculate elapsed time
        elapsed = datetime.now() - self.pipeline_start_time
        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
        
        # Create comprehensive status
        status_lines = []
        status_lines.append("=" * 80)
        status_lines.append(" REDDIT VISA DISCOURSE ANALYSIS PIPELINE - LIVE STATUS")
        status_lines.append("=" * 80)
        status_lines.append(f"⏰ Started: {self.pipeline_start_time.strftime('%H:%M:%S')}")
        status_lines.append(f" Current: {datetime.now().strftime('%H:%M:%S')}")
        status_lines.append(f"⏱  Elapsed: {elapsed_str}")
        status_lines.append("")
        
        # System status
        if system_info['status'] == 'running':
            status_lines.append(" PROCESS STATUS:")
            status_lines.append(f"  PID: {system_info['pid']}")
            status_lines.append(f"  CPU: {system_info['cpu_percent']:.1f}%")
            status_lines.append(f"  Memory: {system_info['memory_mb']:.1f} MB")
        else:
            status_lines.append(" PROCESS STATUS: Not running")
        
        status_lines.append("")
        
        # Output file status
        status_lines.append(" OUTPUT FILES STATUS:")
        
        # Sampled data
        if 'sampled_data' in output_status and 'file' in output_status['sampled_data']:
            data = output_status['sampled_data']
            status_lines.append(f"   Sampled Data: {data['file']} ({data['size_mb']} MB) - {data['modified']}")
        else:
            status_lines.append("  ⏳ Sampled Data: Not yet created")
        
        # Cleaned data
        if 'cleaned_data' in output_status and 'file' in output_status['cleaned_data']:
            data = output_status['cleaned_data']
            status_lines.append(f"   Cleaned Data: {data['file']} ({data['size_mb']} MB) - {data['modified']}")
        else:
            status_lines.append("  ⏳ Cleaned Data: Not yet created")
        
        # Classifier
        if 'classifier' in output_status and 'files' in output_status['classifier']:
            data = output_status['classifier']
            status_lines.append(f"   Classifier: {data['files']} files - {data['latest']}")
        else:
            status_lines.append("  ⏳ Classifier: Not yet created")
        
        status_lines.append("")
        
        # Pipeline steps status
        status_lines.append(" PIPELINE STEPS:")
        steps = [
            ("Data Sampling", "sampled_data" in output_status and 'file' in output_status['sampled_data']),
            ("Data Cleaning", "cleaned_data" in output_status and 'file' in output_status['cleaned_data']),
            ("Descriptive Analysis", False),  # Will be updated when implemented
            ("Annotation", False),
            ("BERT Training", "classifier" in output_status and 'files' in output_status['classifier']),
            ("Evaluation", False),
            ("Temporal Analysis", False),
            ("Visualization", False)
        ]
        
        for step, completed in steps:
            status_icon = "" if completed else "⏳"
            status_lines.append(f"  {status_icon} {step}")
        
        status_lines.append("")
        
        # Estimated completion
        if system_info['status'] == 'running':
            # Rough estimate based on typical pipeline times
            estimated_total = timedelta(hours=2)  # 2 hours total estimate
            remaining = estimated_total - elapsed
            if remaining.total_seconds() > 0:
                status_lines.append(f" ESTIMATED COMPLETION: {remaining}")
            else:
                status_lines.append(" ESTIMATED COMPLETION: Should be completing soon")
        else:
            status_lines.append(" ESTIMATED COMPLETION: Pipeline not running")
        
        status_lines.append("=" * 80)
        
        return "\n".join(status_lines)
    
    def save_comprehensive_status(self):
        """Save comprehensive status to file"""
        status_text = self.generate_comprehensive_status()
        
        # Save to status file
        with open(self.status_file, 'w') as f:
            f.write(status_text)
        
        # Also save to a separate comprehensive file
        with open("comprehensive_pipeline_status.txt", 'w') as f:
            f.write(status_text)
        
        return status_text

def monitor_pipeline_live():
    """Monitor the pipeline and generate live status updates"""
    monitor = LiveStatusMonitor()
    monitor.start_monitoring()
    
    try:
        while True:
            # Generate and save comprehensive status
            status_text = monitor.save_comprehensive_status()
            
            # Print current status
            print(f"\n Status updated at {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 50)
            print(status_text)
            print("=" * 50)
            
            # Wait before next update
            time.sleep(30)  # Update every 30 seconds
            
    except KeyboardInterrupt:
        monitor.update_status(" Monitoring stopped by user")
        print("\n Live monitoring stopped")

if __name__ == "__main__":
    monitor_pipeline_live()
