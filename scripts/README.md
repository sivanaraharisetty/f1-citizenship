# Scripts

This directory contains automation scripts for the Immigration Journey Analyzer.

## 📁 Available Scripts

### 🤖 Monitoring
- **`monitor_and_update.py`** - Automated monitoring and GitHub updates
- **`start_monitoring.sh`** - Easy startup script for monitoring

## 🚀 Usage

### Start Automated Monitoring
```bash
# Start monitoring system
./scripts/start_monitoring.sh

# Check if running
ps aux | grep monitor_and_update.py
```

### Stop Monitoring
```bash
# Stop monitoring
kill $(cat logs/monitoring.pid)
```

## 📊 What Monitoring Does

The monitoring system automatically:

- ✅ **Updates GitHub** every 30 minutes with training progress
- ✅ **Tracks metrics** and performance data
- ✅ **Generates reports** in the `docs/` directory
- ✅ **Logs activities** to `logs/monitoring.log`
- ✅ **Runs unattended** in the background

## 🔧 Configuration

Monitoring settings can be adjusted in the script:

- **Update frequency**: Default 30 minutes
- **Status files**: Auto-generated in `docs/`
- **Log location**: `logs/monitoring.log`
- **PID tracking**: `logs/monitoring.pid`

## 📈 Output Files

The monitoring system generates:

- `docs/REALTIME_STATUS.md` - Live training status
- `docs/TRAINING_ANALYSIS.md` - Performance analysis
- `docs/DASHBOARD.md` - Executive dashboard
- `logs/monitoring.log` - Detailed activity logs

---

*All scripts are designed to run unattended with minimal intervention.*
