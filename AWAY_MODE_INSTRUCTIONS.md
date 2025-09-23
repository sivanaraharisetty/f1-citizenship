# Away Mode Instructions

## Automated Monitoring Setup

Your Immigration Journey Analyzer is now configured for unattended operation with automatic GitHub updates.

## What's Running

### 1. Main Training Process
- **Status**: ACTIVE (2.2% complete, 8,773/382,510 steps)
- **Process**: Background training on 2TB dataset
- **ETA**: ~84 hours remaining
- **Device**: Apple Silicon MPS (GPU Accelerated)

### 2. Automated Monitoring
- **Script**: `monitor_and_update.py`
- **Frequency**: Every 30 minutes
- **Updates**: Real-time progress to GitHub
- **Logs**: `monitoring.log` and `monitor.log`

## How to Start Monitoring (Run This Before Leaving)

```bash
cd /Users/comm-naraharisetty/Documents/f1-citizenship/s3-classifier-project
./start_monitoring.sh
```

This will:
- Start the monitoring script in the background
- Update GitHub every 30 minutes with progress
- Continue running even if you close the terminal
- Save monitoring logs for review

## What Happens Automatically

### Every 30 Minutes:
1. **Reads training logs** for current progress
2. **Updates REALTIME_STATUS.md** with latest metrics
3. **Commits changes** to git with timestamp
4. **Pushes to GitHub** automatically
5. **Logs all activity** for review

### GitHub Updates Include:
- Current training progress percentage
- Processing speed and ETA
- Results file summaries
- Error handling and recovery
- Timestamp of last update

## Monitoring Your Progress Remotely

### Check GitHub Repository:
- **URL**: https://github.com/sivanaraharisetty/f1-citizenship
- **Files to Watch**: 
  - `REALTIME_STATUS.md` (updated every 30 min)
  - `DASHBOARD.md` (executive summary)
  - `TRAINING_ANALYSIS.md` (technical details)

### Check Training Logs:
```bash
# View real-time training progress
tail -f logs/classifier.log

# View monitoring activity
tail -f monitoring.log

# Check if monitoring is running
ps aux | grep monitor_and_update.py
```

## Expected Timeline

### Current Status (as of now):
- **Progress**: 2.2% Complete
- **Current Step**: 8,773/382,510
- **Speed**: 1.23 iterations/second
- **ETA**: ~84 hours

### Completion Schedule:
- **Day 1**: ~15% complete
- **Day 2**: ~30% complete  
- **Day 3**: ~45% complete
- **Day 4**: ~60% complete
- **Day 5**: ~75% complete
- **Day 6**: ~90% complete
- **Day 7**: 100% complete

## Safety Features

### Automatic Recovery:
- **Resume Capability**: Training resumes from last checkpoint
- **Error Handling**: Comprehensive retry logic
- **Progress Persistence**: SQLite-based key tracking
- **State Management**: JSON-based resume state

### Monitoring Alerts:
- **Email Notifications**: Configured for failures
- **Log Rotation**: Prevents disk space issues
- **Memory Management**: Bounded processing
- **Network Resilience**: S3 retry logic

## When You Return

### Check Status:
```bash
# View final results
ls -la results/20250923/

# Check completion status
tail -20 logs/classifier.log

# View monitoring summary
tail -20 monitoring.log
```

### Stop Monitoring (if needed):
```bash
# Stop monitoring script
kill $(cat monitoring.pid)

# Or find and kill manually
ps aux | grep monitor_and_update.py
kill <PID>
```

## Emergency Contacts

### If Training Stops:
1. **Check logs**: `tail -50 logs/classifier.log`
2. **Restart training**: `python -m src.main`
3. **Resume monitoring**: `./start_monitoring.sh`

### If GitHub Updates Fail:
1. **Check git status**: `git status`
2. **Manual push**: `git add . && git commit -m "Manual update" && git push origin main`
3. **Restart monitoring**: `./start_monitoring.sh`

## Repository Status

Your repository will be automatically updated with:
- Real-time training progress
- Model checkpoints and results
- Performance metrics
- Error logs and recovery
- Completion status

**Repository**: https://github.com/sivanaraharisetty/f1-citizenship

---

**Important**: The system is designed to run unattended for days. Training will continue automatically with progress updates every 30 minutes.
