#!/bin/bash
# Unlock /opt/AxonML/crates/ — removes immutable flag.
# Only use when you need to modify the framework itself.
# Re-lock with: sudo ./lock-crates.sh
#
# Usage: sudo ./unlock-crates.sh

set -e

CRATES_DIR="/opt/AxonML/crates"

if [ "$EUID" -ne 0 ]; then
    echo "Error: must run as root (chattr requires CAP_LINUX_IMMUTABLE)"
    echo "Usage: sudo $0"
    exit 1
fi

if [ ! -d "$CRATES_DIR" ]; then
    echo "Error: $CRATES_DIR does not exist"
    exit 1
fi

echo "Unlocking $CRATES_DIR recursively (chattr -i)..."
find "$CRATES_DIR" -exec chattr -i {} + 2>/dev/null
count=$(find "$CRATES_DIR" | wc -l)
echo "  Unlocked $count files/directories"
echo "  Status: READ-WRITE"
echo ""
echo "To re-lock: sudo /opt/AxonML/lock-crates.sh"
