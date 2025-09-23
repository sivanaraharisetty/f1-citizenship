# Immigration Journey Analyzer - Deployment Guide

## Remote System Setup

This guide shows how to deploy and run the Immigration Journey Analyzer on any system with automatic GitHub updates.

## Prerequisites

### System Requirements
- **OS**: macOS, Linux, or Windows with WSL
- **Python**: 3.9+ (recommended 3.11)
- **RAM**: 16GB+ (32GB recommended for 2TB datasets)
- **Storage**: 50GB+ free space (SSD recommended)
- **GPU**: Optional but recommended (CUDA, MPS, or CPU fallback)

### Network Requirements
- **Internet**: Stable connection for S3 access and GitHub updates
- **AWS Credentials**: Configured for S3 data access
- **GitHub Access**: SSH or HTTPS authentication

## Quick Deployment

### 1. Clone Repository
```bash
git clone https://github.com/your-username/immigration-journey-analyzer.git
cd f1-citizenship
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure AWS Credentials
```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

### 4. Configure GitHub Access
```bash
# Option 1: SSH (recommended)
ssh-keygen -t ed25519 -C "your-email@example.com"
# Add public key to GitHub: Settings → SSH and GPG keys

# Option 2: Personal Access Token
git config --global credential.helper store
# Enter GitHub username and PAT when prompted
```

### 5. Customize Configuration
Edit `config/config.yaml`:
```yaml
data:
  s3:
    bucket: "your-bucket-name"
    comments_path: "reddit/comments/"
    posts_path: "reddit/posts/"
  chunking:
    files_per_chunk: 10
    rows_per_chunk: 25000

model:
  parameters:
    learning_rate: 2e-5
    epochs: 1
    batch_size: 32

logging:
  level: "INFO"
  log_file: "logs/classifier.log"
```

## Production Deployment

### 1. System Service (Linux/macOS)
Create systemd service file `/etc/systemd/system/immigration-analyzer.service`:
```ini
[Unit]
Description=Immigration Journey Analyzer
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/f1-citizenship
Environment=PATH=/path/to/f1-citizenship/.venv/bin
ExecStart=/path/to/f1-citizenship/.venv/bin/python -m src.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable immigration-analyzer
sudo systemctl start immigration-analyzer
```

### 2. Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x start_monitoring.sh

CMD ["./start_monitoring.sh"]
```

Build and run:
```bash
docker build -t immigration-analyzer .
docker run -d --name immigration-analyzer \
  -e AWS_ACCESS_KEY_ID="your-key" \
  -e AWS_SECRET_ACCESS_KEY="your-secret" \
  -v $(pwd)/results:/app/results \
  immigration-analyzer
```

### 3. Cloud Deployment (AWS/GCP/Azure)

#### AWS EC2 Setup:
```bash
# Launch EC2 instance (g4dn.xlarge or larger)
# Install dependencies
sudo apt update
sudo apt install python3-pip git

# Clone and setup
git clone https://github.com/your-username/immigration-journey-analyzer.git
cd f1-citizenship
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure AWS credentials
aws configure

# Start training
./start_monitoring.sh
```

#### Google Cloud Platform:
```bash
# Create VM instance
gcloud compute instances create immigration-analyzer \
  --machine-type=n1-standard-4 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --accelerator=type=nvidia-tesla-t4,count=1

# SSH and setup
gcloud compute ssh immigration-analyzer
# Follow same setup as AWS
```

## Remote Monitoring

### 1. GitHub Repository Monitoring
- **URL**: https://github.com/your-username/immigration-journey-analyzer
- **Watch Files**: 
  - `REALTIME_STATUS.md` (updates every 30 min)
  - `DASHBOARD.md` (executive summary)
  - `TRAINING_ANALYSIS.md` (technical details)

### 2. SSH Remote Access
```bash
# Connect to remote system
ssh user@remote-system-ip

# Check training status
tail -f /path/to/f1-citizenship/logs/classifier.log

# Check monitoring
tail -f /path/to/f1-citizenship/monitoring.log

# View results
ls -la /path/to/f1-citizenship/results/YYYYMMDD/
```

### 3. Web Dashboard (Optional)
Create simple web interface `dashboard.py`:
```python
from flask import Flask, render_template, jsonify
import json
import os
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def dashboard():
    # Read status files
    with open('REALTIME_STATUS.md', 'r') as f:
        status = f.read()
    
    return f"""
    <html>
    <head><title>Immigration Journey Analyzer</title></head>
    <body>
        <h1>Immigration Journey Analyzer Dashboard</h1>
        <pre>{status}</pre>
        <p>Last updated: {datetime.now()}</p>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Run dashboard:
```bash
python dashboard.py
# Access at http://your-server-ip:5000
```

## Automated Monitoring Setup

### 1. Start Monitoring
```bash
cd /path/to/f1-citizenship
./start_monitoring.sh
```

### 2. Verify Monitoring
```bash
# Check if monitoring is running
ps aux | grep monitor_and_update.py

# Check monitoring logs
tail -f monitoring.log

# Check PID file
cat monitoring.pid
```

### 3. Stop Monitoring
```bash
# Stop monitoring script
kill $(cat monitoring.pid)

# Or find and kill manually
ps aux | grep monitor_and_update.py
kill <PID>
```

## Troubleshooting

### Common Issues

#### 1. Training Stops
```bash
# Check logs
tail -50 logs/classifier.log

# Restart training
python -m src.main

# Restart monitoring
./start_monitoring.sh
```

#### 2. GitHub Updates Fail
```bash
# Check git status
git status

# Manual push
git add . && git commit -m "Manual update" && git push origin main

# Restart monitoring
./start_monitoring.sh
```

#### 3. Memory Issues
```bash
# Reduce batch size in config.yaml
model:
  parameters:
    batch_size: 8  # Reduce from 16

# Reduce chunk size
data:
  chunking:
    rows_per_chunk: 10000  # Reduce from 25000
```

#### 4. S3 Access Issues
```bash
# Test S3 access
aws s3 ls s3://your-bucket-name/

# Check credentials
aws sts get-caller-identity

# Update credentials
aws configure
```

## Performance Optimization

### 1. Hardware Optimization
- **GPU**: Use CUDA-enabled GPU for faster training
- **Memory**: Increase RAM for larger batch sizes
- **Storage**: Use SSD for checkpoint I/O
- **Network**: Ensure stable internet connection

### 2. Software Optimization
```yaml
# config/config.yaml optimizations
data:
  chunking:
    rows_per_chunk: 50000  # Increase for better GPU utilization
    files_per_chunk: 20    # Increase for fewer S3 calls

model:
  parameters:
    batch_size: 32         # Increase for better GPU utilization
    epochs: 1              # Keep low for streaming training
```

### 3. Monitoring Optimization
```bash
# Increase monitoring frequency
# Edit monitor_and_update.py, change sleep(1800) to sleep(900)  # 15 minutes

# Add more detailed logging
# Edit config.yaml
logging:
  level: "DEBUG"
```

## Security Considerations

### 1. Credential Management
```bash
# Use environment variables
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"

# Or use AWS IAM roles (recommended for EC2)
```

### 2. Network Security
```bash
# Firewall rules (if needed)
sudo ufw allow 22    # SSH
sudo ufw allow 5000  # Web dashboard (if used)
sudo ufw enable
```

### 3. Data Protection
```bash
# Encrypt sensitive data
gpg --symmetric --cipher-algo AES256 config.yaml

# Use secure file permissions
chmod 600 config.yaml
```

## Backup and Recovery

### 1. Automated Backups
```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "backup_${DATE}.tar.gz" results/ logs/ config/
aws s3 cp "backup_${DATE}.tar.gz" s3://your-backup-bucket/
EOF

chmod +x backup.sh

# Schedule daily backups
echo "0 2 * * * /path/to/backup.sh" | crontab -
```

### 2. Recovery Process
```bash
# Restore from backup
aws s3 cp s3://your-backup-bucket/backup_YYYYMMDD_HHMMSS.tar.gz .
tar -xzf backup_YYYYMMDD_HHMMSS.tar.gz

# Resume training
python -m src.main
```

## Monitoring and Alerts

### 1. Email Notifications
Configure in `src/main.py`:
```python
alert_sender = "your-email@domain.com"
alert_recipient = "manager@domain.com"
```

### 2. Slack Integration
```python
# Add to monitor_and_update.py
import requests

def send_slack_alert(message):
    webhook_url = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
    requests.post(webhook_url, json={"text": message})
```

### 3. Health Checks
```bash
# Create health check script
cat > health_check.sh << 'EOF'
#!/bin/bash
if ! pgrep -f "python -m src.main" > /dev/null; then
    echo "Training stopped!" | mail -s "Alert" your-email@domain.com
fi
EOF

chmod +x health_check.sh
echo "*/5 * * * * /path/to/health_check.sh" | crontab -
```

---

**Repository**: https://github.com/your-username/immigration-journey-analyzer  
**Support**: Check `AWAY_MODE_INSTRUCTIONS.md` for detailed monitoring  
**Documentation**: All files are automatically updated every 30 minutes
