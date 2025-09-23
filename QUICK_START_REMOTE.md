# Quick Start - Remote Deployment

## 5-Minute Setup on Any System

### 1. Clone and Setup
```bash
git clone https://github.com/sivanaraharisetty/f1-citizenship.git
cd f1-citizenship
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Access
```bash
# AWS credentials
aws configure
# OR
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"

# GitHub access
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

### 3. Start Training
```bash
./start_monitoring.sh
```

That's it! The system will:
- Start training on 2TB dataset
- Update GitHub every 30 minutes
- Run unattended for days
- Resume automatically if interrupted

## Monitor Progress

### GitHub Repository
- **URL**: https://github.com/sivanaraharisetty/f1-citizenship
- **Files to Watch**: `REALTIME_STATUS.md` (updates every 30 min)

### Local Monitoring
```bash
# Check training progress
tail -f logs/classifier.log

# Check monitoring activity
tail -f monitoring.log

# View results
ls -la results/YYYYMMDD/
```

## Expected Timeline
- **Day 1**: ~15% complete
- **Day 2**: ~30% complete
- **Day 3**: ~45% complete
- **Day 4**: ~60% complete
- **Day 5**: ~75% complete
- **Day 6**: ~90% complete
- **Day 7**: 100% complete

## Troubleshooting

### If Training Stops
```bash
python -m src.main
```

### If GitHub Updates Fail
```bash
git add . && git commit -m "Manual update" && git push origin main
```

### If Monitoring Stops
```bash
./start_monitoring.sh
```

---

**Full Documentation**: See `DEPLOYMENT_GUIDE.md` for advanced setup options
