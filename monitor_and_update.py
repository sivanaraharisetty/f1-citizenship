#!/usr/bin/env python3
"""
Automated monitoring and GitHub update script for Immigration Journey Analyzer.
Updates GitHub with real-time progress every 30 minutes.
"""

import os
import json
import time
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_training_progress():
    """Extract current training progress from logs."""
    try:
        # Read the latest log entries
        log_file = "logs/classifier.log"
        if not os.path.exists(log_file):
            return None
            
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Look for progress indicators
        progress_info = {}
        for line in reversed(lines[-50:]):  # Check last 50 lines
            if "Processing chunk" in line:
                progress_info['chunk_info'] = line.strip()
            elif "Finished chunk" in line:
                progress_info['completion'] = line.strip()
            elif "ETA" in line:
                progress_info['eta'] = line.strip()
            elif "Speed" in line:
                progress_info['speed'] = line.strip()
        
        return progress_info
    except Exception as e:
        logger.error(f"Error reading progress: {e}")
        return None

def get_results_summary():
    """Get summary of current results."""
    try:
        results_dir = f"results/{datetime.now().strftime('%Y%m%d')}"
        if not os.path.exists(results_dir):
            return None
            
        summary = {
            'timestamp': datetime.now().isoformat(),
            'results_dir': results_dir,
            'files': []
        }
        
        # List result files
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file.endswith(('.json', '.sqlite', '.log')):
                    file_path = os.path.join(root, file)
                    stat = os.stat(file_path)
                    summary['files'].append({
                        'name': file,
                        'path': file_path,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        return summary
    except Exception as e:
        logger.error(f"Error getting results summary: {e}")
        return None

def update_status_file():
    """Update the real-time status file."""
    try:
        progress = get_training_progress()
        results = get_results_summary()
        
        status_content = f"""# Real-Time Training Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

## Current Status
- **Training**: ACTIVE
- **Last Update**: {datetime.now().isoformat()}
- **Progress**: {progress.get('chunk_info', 'Processing...') if progress else 'Initializing...'}
- **Completion**: {progress.get('completion', 'In Progress') if progress else 'Starting...'}
- **Speed**: {progress.get('speed', 'Calculating...') if progress else 'Initializing...'}
- **ETA**: {progress.get('eta', 'Calculating...') if progress else 'Initializing...'}

## Results Summary
"""
        
        if results:
            status_content += f"- **Results Directory**: {results['results_dir']}\n"
            status_content += f"- **Files Generated**: {len(results['files'])}\n"
            status_content += f"- **Total Size**: {sum(f['size'] for f in results['files']):,} bytes\n"
        else:
            status_content += "- **Results**: Initializing...\n"
        
        status_content += f"""
## Monitoring
- **Script**: monitor_and_update.py
- **Update Frequency**: Every 30 minutes
- **Log File**: monitor.log
- **Repository**: https://github.com/your-username/immigration-journey-analyzer

---
*Auto-generated at {datetime.now().isoformat()}*
"""
        
        with open('REALTIME_STATUS.md', 'w') as f:
            f.write(status_content)
        
        logger.info("Status file updated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error updating status file: {e}")
        return False

def commit_and_push():
    """Commit changes and push to GitHub."""
    try:
        # Add all changes
        subprocess.run(['git', 'add', '.'], check=True, capture_output=True)
        
        # Commit with timestamp
        commit_msg = f"Auto-update: Training progress at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True, capture_output=True)
        
        # Push to GitHub
        result = subprocess.run(['git', 'push', 'origin', 'main'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Successfully pushed to GitHub")
            return True
        else:
            logger.error(f"Git push failed: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Error in commit/push: {e}")
        return False

def main():
    """Main monitoring loop."""
    logger.info("Starting automated monitoring and GitHub updates")
    
    while True:
        try:
            logger.info("Updating status and pushing to GitHub...")
            
            # Update status file
            if update_status_file():
                logger.info("Status file updated")
            else:
                logger.error("Failed to update status file")
            
            # Commit and push to GitHub
            if commit_and_push():
                logger.info("Successfully pushed to GitHub")
            else:
                logger.error("Failed to push to GitHub")
            
            # Wait 30 minutes before next update
            logger.info("Waiting 30 minutes until next update...")
            time.sleep(1800)  # 30 minutes
            
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.info("Waiting 5 minutes before retry...")
            time.sleep(300)  # 5 minutes

if __name__ == "__main__":
    main()
