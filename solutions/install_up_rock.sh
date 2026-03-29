#!/bin/bash
# install_up_rock.sh — Reproducible ROCK Admin setup
# Sets up and starts the ROCK Admin service (required before ROLL agentic pipeline)
set -euo pipefail

ROCK_DIR="/home/ubuntu/ALE-latest/ROCK-personal"

echo "=== ROCK Admin Setup ==="

cd ${ROCK_DIR}

# Install dependencies (uses existing .venv with Python 3.11)
echo "Installing ROCK dependencies..."
uv sync --all-extras --all-groups

# Start ROCK Admin service
echo "Starting ROCK Admin..."
# Option 1: Use rock CLI (if installed globally)
# rock admin start

# Option 2: Use uv to run from venv
uv run rock admin start &
ROCK_PID=$!

# Wait for startup
echo "Waiting for ROCK Admin to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8080/ > /dev/null 2>&1; then
        echo "ROCK Admin is running at http://localhost:8080"
        break
    fi
    sleep 2
done

# Verify
RESPONSE=$(curl -s http://localhost:8080/)
echo "ROCK response: ${RESPONSE}"

if echo "${RESPONSE}" | grep -q "ROCK"; then
    echo "=== ROCK Admin setup complete ==="
else
    echo "ERROR: ROCK Admin did not start correctly"
    exit 1
fi

echo ""
echo "ROCK Admin PID: ${ROCK_PID}"
echo "To stop: kill ${ROCK_PID}"
echo "To check: curl http://localhost:8080/"
