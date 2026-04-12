#!/bin/bash
#
# AxonML - ML Framework Database Initialization
# Database Initialization Script for Aegis-DB
#
# This script initializes the Aegis-DB schema and creates the default admin user.
# Run this script when setting up a new instance or after clearing the database.
#
# NOTE: The AxonML server also creates these collections on startup via
# Schema::init(). This script's main value is seeding the admin user with a
# known password BEFORE the first server start, so the random-password path
# in main.rs is skipped and you can log in immediately.
#
# Usage: ./AxonML_DB_Init.sh [--with-user]
#
# Options:
#   --with-user    Also create the DevOps admin user
#
# Environment Variables:
#   AEGIS_URL      Aegis-DB URL (default: http://127.0.0.1:9090)
#   AEGIS_USER     Aegis-DB username for authentication
#   AEGIS_PASS     Aegis-DB password for authentication
#

set -e

# Configuration — default port matches Aegis-DB's own default (9090)
AEGIS_URL="${AEGIS_URL:-http://127.0.0.1:9090}"
AEGIS_USER="${AEGIS_USER:-}"
AEGIS_PASS="${AEGIS_PASS:-}"
API_BASE="$AEGIS_URL/api/v1/documents/collections"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# All 8 AxonML collections (must match axonml-server/src/db/schema.rs)
COLLECTIONS=(
    "axonml_users"
    "axonml_runs"
    "axonml_models"
    "axonml_model_versions"
    "axonml_endpoints"
    "axonml_datasets"
    "axonml_notebooks"
    "axonml_checkpoints"
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
            ;;
    esac
done

# Check Aegis-DB connection
echo -e "${YELLOW}Checking Aegis-DB connection...${NC}"
if ! curl -sf "$AEGIS_URL/health" > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Cannot connect to Aegis-DB at $AEGIS_URL${NC}"
    echo "Make sure Aegis-DB is running:"
    echo "  aegis-server --port 9090 --data-dir /tmp/aegis-data"
    exit 1
fi
echo -e "${GREEN}✓ Connected to Aegis-DB at $AEGIS_URL${NC}"
echo ""

# Authenticate with Aegis-DB
AUTH_HEADER=""
if [ -n "$AEGIS_USER" ] && [ -n "$AEGIS_PASS" ]; then
    echo -e "${YELLOW}Authenticating with Aegis-DB...${NC}"
    auth_response=$(curl -sf -X POST "$AEGIS_URL/api/v1/auth/login" \
        -H "Content-Type: application/json" \
        -d "{\"username\": \"$AEGIS_USER\", \"password\": \"$AEGIS_PASS\"}" 2>/dev/null) || {
        echo -e "${RED}ERROR: Authentication failed. Check AEGIS_USER / AEGIS_PASS.${NC}"
        exit 1
    }
    # Extract token from JSON response (simple grep — works for {"token":"..."})
    token=$(echo "$auth_response" | grep -oP '"token"\s*:\s*"\K[^"]+' 2>/dev/null || true)
    if [ -n "$token" ]; then
        AUTH_HEADER="Authorization: Bearer $token"
        echo -e "${GREEN}✓ Authenticated${NC}"
    else
        echo -e "${YELLOW}⚠ Could not extract token — proceeding without auth (may fail if auth is required)${NC}"
    fi
    echo ""
else
    echo -e "${YELLOW}⚠ AEGIS_USER / AEGIS_PASS not set — proceeding without auth${NC}"
    echo "  Set these env vars if Aegis-DB requires authentication."
    echo ""
fi

# Helper: curl with optional auth header
aegis_curl() {
    if [ -n "$AUTH_HEADER" ]; then
        curl -sf -H "$AUTH_HEADER" "$@"
    else
        curl -sf "$@"
    fi
}

# Create collections
echo -e "${YELLOW}Creating AxonML collections (${#COLLECTIONS[@]} total)...${NC}"
echo "────────────────────────────────────────"

for collection in "${COLLECTIONS[@]}"; do
    echo -n "  Creating $collection... "

    response=$(aegis_curl -X POST "$API_BASE" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$collection\"}" 2>&1) || true

    # Check for success or already-exists
    if echo "$response" | grep -qiE '"error".*already.exists|"error".*duplicate'; then
        echo -e "${YELLOW}exists${NC}"
    elif [ -z "$response" ]; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${GREEN}✓${NC}"
    fi
done

echo "────────────────────────────────────────"
echo -e "${GREEN}✓ All collections processed${NC}"
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

if aegis_curl -X POST "$API_BASE/axonml_users/documents" \
    -H "Content-Type: application/json" \
    -d "$admin_doc" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Default admin user created${NC}"
else
    echo -e "${YELLOW}⚠ Admin user may already exist (skipped)${NC}"
fi
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
        "email": "DevOps@AutomataNexus.com",
        "name": "DevOps Admin",
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

    if aegis_curl -X POST "$API_BASE/axonml_users/documents" \
        -H "Content-Type: application/json" \
        -d "$devops_doc" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ DevOps admin user created${NC}"
    else
        echo -e "${YELLOW}⚠ DevOps user may already exist (skipped)${NC}"
    fi
    echo "    Email: DevOps@AutomataNexus.com"
    echo "    Name: DevOps Admin"
    echo ""
fi

# Summary
echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                      Initialization Complete                           ║"
echo "╠════════════════════════════════════════════════════════════════════════╣"
echo "║  Collections: ${#COLLECTIONS[@]} processed                                           ║"
echo "║  Admin User:  admin@axonml.local / admin                               ║"
if [ "$WITH_USER" = true ]; then
echo "║  DevOps User: DevOps@AutomataNexus.com                                  ║"
fi
echo "║                                                                        ║"
echo "║  Start AxonML Server:                                                  ║"
echo "║    cargo run -p axonml-server -- --port 3000                           ║"
echo "║                                                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
