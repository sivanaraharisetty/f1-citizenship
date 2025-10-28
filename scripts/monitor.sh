#!/bin/bash
# Production Sampling Monitor for macOS
# Alternative to the 'watch' command

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display status
display_status() {
    clear
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${GREEN} PRODUCTION SAMPLING STATUS${NC}"
    echo -e "${YELLOW}⏰ $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${BLUE}============================================================${NC}"
    
    echo -e "\n${GREEN} RUNNING PROCESSES:${NC}"
    ps aux | grep production_sampling | grep -v grep || echo "  No production sampling processes found"
    
    echo -e "\n${GREEN} LATEST LOG ENTRIES:${NC}"
    if [ -f "sampled_data/sampling_log.txt" ]; then
        tail -5 sampled_data/sampling_log.txt | sed 's/^/  /'
    else
        echo "  No log file found"
    fi
    
    echo -e "\n${GREEN} BATCH FILES:${NC}"
    if [ -d "sampled_data/batches" ]; then
        ls -la sampled_data/batches/ 2>/dev/null | tail -5 | sed 's/^/  /' || echo "  No batch files yet"
    else
        echo "  No batches directory found"
    fi
    
    echo -e "\n${GREEN} OUTPUT FILES:${NC}"
    ls -la sampled_data/*.parquet 2>/dev/null | tail -3 | sed 's/^/  /' || echo "  No parquet files found"
    
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${YELLOW}Press Ctrl+C to exit${NC}"
}

# Check if running in continuous mode or once
if [ "$1" = "--once" ]; then
    display_status
    exit 0
fi

# Continuous monitoring
echo "Starting production sampling monitor..."
echo "Press Ctrl+C to exit"
sleep 2

while true; do
    display_status
    sleep 5
done
