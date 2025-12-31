#!/bin/bash
# Topo-tessellate Runner Script
# Runs the terrain generator inside the Docker container
#
# Usage:
#   ./run.sh --config configs/my_config.yaml --topo-dir ./topo --output-dir ./outputs
#
# All directories will be mounted into the container.

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
Topo-Tessellate Terrain Generator

Usage: $0 --config <config.yaml> --topo-dir <directory> --output-dir <directory> [OPTIONS]

Required:
  --config <file>       Path to YAML configuration file
  --topo-dir <dir>      Directory containing elevation data (GeoTIFF files)
  --output-dir <dir>    Directory to save generated STL files

Options:
  --job <name>          Run only the named job (for multi-job configs)
  -h, --help            Show this help message

Example:
  $0 --config configs/sf_fidi.yaml --topo-dir ./topo --output-dir ./outputs

EOF
    exit 0
}

# Check for Docker
if ! command -v docker &> /dev/null; then
    error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Parse arguments
CONFIG=""
TOPO_DIR=""
OUTPUT_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --topo-dir)
            TOPO_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --job)
            EXTRA_ARGS+=("--job" "$2")
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CONFIG" ]]; then
    error "Missing required argument: --config"
    usage
fi

if [[ -z "$TOPO_DIR" ]]; then
    error "Missing required argument: --topo-dir"
    usage
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    error "Missing required argument: --output-dir"
    usage
fi

# Check config file exists
if [[ ! -f "$CONFIG" ]]; then
    error "Config file not found: $CONFIG"
    exit 1
fi

# Check topo directory exists
if [[ ! -d "$TOPO_DIR" ]]; then
    error "Topo directory not found: $TOPO_DIR"
    error "Run download-dem.sh first to download elevation data."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Convert to absolute paths
CONFIG_ABS="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
TOPO_ABS="$(cd "$TOPO_DIR" && pwd)"
OUTPUT_ABS="$(cd "$OUTPUT_DIR" && pwd)"

# Get config directory and filename
CONFIG_DIR="$(dirname "$CONFIG_ABS")"
CONFIG_FILE="$(basename "$CONFIG_ABS")"

# Always pull the latest image
info "Pulling latest container image..."
docker pull "$IMAGE" --quiet

info "Generating terrain model..."
info "  Config: $CONFIG"
info "  Topo:   $TOPO_DIR"
info "  Output: $OUTPUT_DIR"

# Run the container with mounts
docker run --rm \
    -v "${CONFIG_DIR}:/app/configs:ro" \
    -v "${TOPO_ABS}:/app/topo:ro" \
    -v "${OUTPUT_ABS}:/app/outputs" \
    "$IMAGE" \
    --config "/app/configs/${CONFIG_FILE}" \
    --outdir /app/outputs \
    "${EXTRA_ARGS[@]}"

info "Done! Check $OUTPUT_DIR for generated STL files."
