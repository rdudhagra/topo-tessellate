#!/bin/bash
# Interactive configuration creator for Topo-Tessellate
# This script runs the create_config.py script inside the Docker container
#
# Usage:
#   ./create-config.sh configs/my_config.yaml
#
# The config file will be created at the specified path.

set -e

IMAGE="ghcr.io/rdudhagra/topo-tessellate:latest"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $1" >&2; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1" >&2; }
error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

usage() {
    cat <<EOF
Create a Topo-Tessellate configuration file interactively

Usage: $0 <config_file.yaml>

Arguments:
  config_file.yaml    Path where the configuration file will be created

Example:
  $0 configs/my_terrain.yaml
EOF
    exit 0
}

# Check for Docker
if ! command -v docker &> /dev/null; then
    error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Parse arguments
if [[ $# -eq 0 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage
fi

CONFIG_PATH="$1"

# Ensure the parent directory exists
CONFIG_DIR="$(dirname "$CONFIG_PATH")"
CONFIG_FILE="$(basename "$CONFIG_PATH")"

if [[ -n "$CONFIG_DIR" ]] && [[ "$CONFIG_DIR" != "." ]]; then
    mkdir -p "$CONFIG_DIR"
fi

# Convert to absolute path
if [[ "$CONFIG_DIR" == "." ]]; then
    CONFIG_DIR="$(pwd)"
else
    CONFIG_DIR="$(cd "$CONFIG_DIR" && pwd)"
fi

# Always pull the latest image
info "Pulling latest container image..."
docker pull "$IMAGE" --quiet

info "Starting interactive configuration wizard..."

# Run the config creator inside the container with TTY for interactive input
docker run --rm -it \
    -v "${CONFIG_DIR}:/app/output_configs" \
    --entrypoint python \
    "$IMAGE" \
    scripts/create_config.py \
    "/app/output_configs/${CONFIG_FILE}"

info "Configuration file created: $CONFIG_PATH"

