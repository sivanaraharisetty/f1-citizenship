#!/bin/bash
# Start automated monitoring and GitHub updates

echo "Starting Immigration Journey Analyzer monitoring system..."

# Change to project directory
cd /Users/comm-naraharisetty/Documents/f1-citizenship/s3-classifier-project

# Activate virtual environment
source .venv/bin/activate

# Start monitoring script in background
nohup python monitor_and_update.py > monitoring.log 2>&1 &

# Get the PID
MONITOR_PID=$!
echo "Monitoring started with PID: $MONITOR_PID"
echo "PID saved to monitoring.pid"

# Save PID for later reference
echo $MONITOR_PID > monitoring.pid

echo "Monitoring system is now running in the background."
echo "It will update GitHub every 30 minutes with training progress."
echo "Check monitoring.log for detailed logs."
echo "To stop monitoring: kill \$(cat monitoring.pid)"

# Show current status
echo ""
echo "Current training status:"
tail -5 logs/classifier.log 2>/dev/null || echo "Training logs not yet available"

echo ""
echo "Monitoring will continue even if you close this terminal."
echo "GitHub will be updated automatically every 30 minutes."
