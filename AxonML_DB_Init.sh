#!/bin/bash
#
# AxonML - ML Framework Database Initialization
# Database Initialization Script for Aegis-DB
#
# This script initializes the Aegis-DB schema and creates the default admin user.
# Run this script when setting up a new instance or after clearing the database.
#
# Usage: ./AxonML_DB_Init.sh [--with-user]
#
# Options:
#   --with-user    Also create the DevOps admin user
#
# Environment Variables:
#   AEGIS_URL      Aegis-DB URL (default: http://127.0.0.1:7001)
#

set -e

# Configuration
AEGIS_URL="${AEGIS_URL:-http://127.0.0.1:7001}"
API_BASE="$AEGIS_URL/api/v1/documents/collections"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# AxonML Collections
COLLECTIONS=(
    "axonml_users"
    "axonml_runs"
    "axonml_models"
    "axonml_model_versions"
    "axonml_endpoints"
    "axonml_datasets"
)

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                        ║"
echo "║      █████╗ ██╗  ██╗ ██████╗ ███╗   ██╗███╗   ███╗██╗                  ║"
echo "║     ██╔══██╗╚██╗██╔╝██╔═══██╗████╗  ██║████╗ ████║██║                  ║"
echo "║     ███████║ ╚███╔╝ ██║   ██║██╔██╗ ██║██╔████╔██║██║                  ║"
echo "║     ██╔══██║ ██╔██╗ ██║   ██║██║╚██╗██║██║╚██╔╝██║██║                  ║"
echo "║     ██║  ██║██╔╝ ██╗╚██████╔╝██║ ╚████║██║ ╚═╝ ██║███████╗            ║"
echo "║     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝╚══════╝            ║"
echo "║                                                                        ║"
echo "║                   Database Initialization Script                       ║"
echo "║                                                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Parse arguments
WITH_USER=false
for arg in "$@"; do
    case $arg in
        --with-user)
            WITH_USER=true
            shift
            ;;
    esac
done

# Check Aegis-DB connection
echo -e "${YELLOW}Checking Aegis-DB connection...${NC}"
if ! curl -s "$AEGIS_URL/health" > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Cannot connect to Aegis-DB at $AEGIS_URL${NC}"
    echo "Make sure Aegis-DB is running:"
    echo "  aegis-server --port 7001 --data-dir /tmp/aegis-data"
    exit 1
fi
echo -e "${GREEN}✓ Connected to Aegis-DB at $AEGIS_URL${NC}"
echo ""

# Create collections
echo -e "${YELLOW}Creating AxonML collections...${NC}"
echo "────────────────────────────────────────"

for collection in "${COLLECTIONS[@]}"; do
    echo -n "  Creating $collection... "

    response=$(curl -s -X POST "$API_BASE" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$collection\"}" 2>/dev/null)

    echo -e "${GREEN}✓${NC}"
done

echo "────────────────────────────────────────"
echo -e "${GREEN}✓ All collections created${NC}"
echo ""

# Create default admin user
echo -e "${YELLOW}Creating default admin user...${NC}"

timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.000000000Z")

admin_doc=$(cat <<EOF
{
    "document": {
        "_id": "admin",
        "id": "admin",
        "email": "admin@axonml.local",
        "name": "Administrator",
        "password_hash": "\$argon2id\$v=19\$m=19456,t=2,p=1\$BXWl9FZevLFrMidtrqmceA\$cCA7K8R4TQZagGAX6uWml2fwm9VzyzWm3aFgX8oJU/0",
        "role": "admin",
        "mfa_enabled": false,
        "totp_secret": null,
        "webauthn_credentials": [],
        "recovery_codes": [],
        "email_pending": false,
        "email_verified": true,
        "verification_token": null,
        "created_at": "$timestamp",
        "updated_at": "$timestamp"
    }
}
EOF
)

curl -s -X POST "$API_BASE/axonml_users/documents" \
    -H "Content-Type: application/json" \
    -d "$admin_doc" > /dev/null 2>&1

echo -e "${GREEN}✓ Default admin user created${NC}"
echo "    Email: admin@axonml.local"
echo "    Password: admin"
echo ""

# Create DevOps user if requested
if [ "$WITH_USER" = true ]; then
    echo -e "${YELLOW}Creating DevOps admin user...${NC}"

    user_id=$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid)

    devops_doc=$(cat <<EOF
{
    "document": {
        "_id": "$user_id",
        "id": "$user_id",
        "email": "DevOps@automatanexus.com",
        "name": "Andrew Jewell",
        "password_hash": "\$argon2id\$v=19\$m=19456,t=2,p=1\$acr9WUuS7lg2yoi8AHZAOQ\$JsbYql+uEabmalV21GLetVjDZ3Q4MImyqXEx77nOlfM",
        "role": "admin",
        "mfa_enabled": false,
        "totp_secret": null,
        "webauthn_credentials": [],
        "recovery_codes": [],
        "email_pending": false,
        "email_verified": true,
        "verification_token": null,
        "created_at": "$timestamp",
        "updated_at": "$timestamp"
    }
}
EOF
)

    curl -s -X POST "$API_BASE/axonml_users/documents" \
        -H "Content-Type: application/json" \
        -d "$devops_doc" > /dev/null 2>&1

    echo -e "${GREEN}✓ DevOps admin user created${NC}"
    echo "    Email: DevOps@automatanexus.com"
    echo "    Name: Andrew Jewell"
    echo ""
fi

# Summary
echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                      Initialization Complete                           ║"
echo "╠════════════════════════════════════════════════════════════════════════╣"
echo "║  Collections: ${#COLLECTIONS[@]} created                                            ║"
echo "║  Admin User:  admin@axonml.local / admin                               ║"
if [ "$WITH_USER" = true ]; then
echo "║  DevOps User: DevOps@automatanexus.com                                 ║"
fi
echo "║                                                                        ║"
echo "║  Start AxonML Server:                                                  ║"
echo "║    cargo run -p axonml-server -- --port 3021                           ║"
echo "║                                                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
