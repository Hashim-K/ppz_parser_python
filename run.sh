#!/bin/bash

# Usage: ./run.sh [--plot|--events] [--force] [--debug|--info|--warning|--error] [logfile]
# Default logfile
LOGFILE="25_07_09__15_42_36.log"

# Default log level
LOG_LEVEL="debug"

# Parse arguments
PLOT_MODE=false
EVENTS_MODE=false
FORCE_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --plot)
            PLOT_MODE=true
            shift
            ;;
        --events)
            EVENTS_MODE=true
            shift
            ;;
        --force)
            FORCE_MODE=true
            shift
            ;;
        --debug)
            LOG_LEVEL="debug"
            shift
            ;;
        --info)
            LOG_LEVEL="info"
            shift
            ;;
        --warning)
            LOG_LEVEL="warning"
            shift
            ;;
        --error)
            LOG_LEVEL="error"
            shift
            ;;
        *.log)
            LOGFILE="$1"
            shift
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: ./run.sh [--plot|--events] [--force] [--debug|--info|--warning|--error] [logfile]"
            exit 1
            ;;
    esac
done

# Execute based on flags
if [ "$PLOT_MODE" = true ] && [ "$EVENTS_MODE" = true ]; then
    echo "Running both plotting and event analysis..."
    .venv/bin/python main_plot.py "$LOGFILE"
    echo "================================"
    if [ "$FORCE_MODE" = true ]; then
        .venv/bin/python main_events.py "$LOGFILE" --force --log_level "$LOG_LEVEL"
    else
        .venv/bin/python main_events.py "$LOGFILE" --log_level "$LOG_LEVEL"
    fi
elif [ "$PLOT_MODE" = true ]; then
    .venv/bin/python main_plot.py "$LOGFILE"
elif [ "$EVENTS_MODE" = true ]; then
    if [ "$FORCE_MODE" = true ]; then
        .venv/bin/python main_events.py "$LOGFILE" --force --log_level "$LOG_LEVEL"
    else
        .venv/bin/python main_events.py "$LOGFILE" --log_level "$LOG_LEVEL"
    fi
else
    .venv/bin/python -m parser.main "$LOGFILE"
fi