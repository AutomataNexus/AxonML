#!/bin/bash
# Lock /opt/AxonML/crates/ as immutable (read-only at filesystem level).
# Prevents ALL writes including sudo/rm -rf.
# Unlock with: ./unlock-crates.sh
#
# Usage: sudo ./lock-crates.sh

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

echo "Locking $CRATES_DIR recursively (chattr +i)..."
find "$CRATES_DIR" -exec chattr +i {} + 2>/dev/null
count=$(find "$CRATES_DIR" | wc -l)
echo "  Locked $count files/directories"
echo "  Status: READ-ONLY (immutable)"
echo ""
echo "To unlock: sudo /opt/AxonML/unlock-crates.sh"
