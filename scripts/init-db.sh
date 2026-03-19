#!/bin/bash
# AxonML Database Initialization Script
# Creates all required collections and the devops admin user

set -e

AEGIS_URL="${AEGIS_URL:-http://127.0.0.1:7001}"
API="${AEGIS_URL}/api/v1"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           AxonML Database Initialization                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Aegis-DB URL: $AEGIS_URL"
echo ""

# Check if Aegis is healthy
echo "Checking Aegis-DB health..."
HEALTH=$(curl -s "$AEGIS_URL/health" | grep -o '"status":"healthy"' || true)
if [ -z "$HEALTH" ]; then
    echo "ERROR: Aegis-DB is not responding at $AEGIS_URL"
    exit 1
fi
echo "  Aegis-DB is healthy"
echo ""

# Collections to create
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

echo "Creating collections..."
for COLLECTION in "${COLLECTIONS[@]}"; do
    RESULT=$(curl -s -X POST "${API}/documents/collections" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$COLLECTION\"}" 2>/dev/null)

    if echo "$RESULT" | grep -q "already exists" 2>/dev/null; then
        echo "  [exists] $COLLECTION"
    else
        echo "  [created] $COLLECTION"
    fi
done
echo ""

# Create default admin user
echo "Creating admin user..."
ADMIN_RESULT=$(curl -s -X POST "${API}/documents/collections/axonml_users/documents" \
    -H "Content-Type: application/json" \
    -d '{
        "id": "admin",
        "document": {
            "id": "admin",
            "email": "admin@axonml.local",
            "name": "Administrator",
            "password_hash": "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$RdescudvJCsgt3ub+b+dWRWJTmaaJObG",
            "role": "admin",
            "mfa_enabled": false,
            "totp_secret": null,
            "webauthn_credentials": [],
            "recovery_codes": [],
            "email_pending": false,
            "email_verified": true,
            "verification_token": null,
            "created_at": "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'",
            "updated_at": "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'"
        }
    }' 2>/dev/null)
echo "  admin@axonml.local: $ADMIN_RESULT"

# Create DevOps admin user
if [ "$1" = "--with-user" ]; then
    echo ""
    echo "Creating DevOps admin user..."
    DEVOPS_RESULT=$(curl -s -X POST "${API}/documents/collections/axonml_users/documents" \
        -H "Content-Type: application/json" \
        -d '{
            "id": "devops",
            "document": {
                "id": "devops",
                "email": "devops@example.com",
                "name": "DevOps Admin",
                "password_hash": "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$RdescudvJCsgt3ub+b+dWRWJTmaaJObG",
                "role": "admin",
                "mfa_enabled": false,
                "totp_secret": null,
                "webauthn_credentials": [],
                "recovery_codes": [],
                "email_pending": false,
                "email_verified": true,
                "verification_token": null,
                "created_at": "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'",
                "updated_at": "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'"
            }
        }' 2>/dev/null)
    echo "  devops@example.com: $DEVOPS_RESULT"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║           Database Initialization Complete                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Default credentials:"
echo "  Admin:  admin@axonml.local / admin"
if [ "$1" = "--with-user" ]; then
    echo "  DevOps: DevOps@AutomataNexus.com / admin"
fi
echo ""
