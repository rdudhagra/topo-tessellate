#!/bin/bash
# Download DEM (Digital Elevation Model) data from USGS National Map
# This script runs the download_dem.py script inside the Docker container
#
# Usage:
#   ./download-dem.sh --config configs/my_config.yaml --topo-dir ./topo
#
# The topo directory will be mounted into the container and populated with
# downloaded GeoTIFF files.

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
Download DEM data from USGS National Map

Usage: $0 --config <config.yaml> --topo-dir <directory> [OPTIONS]

Required:
  --config <file>      Path to YAML configuration file
  --topo-dir <dir>     Directory to store downloaded DEM files

Options:
  --dry-run            Show what would be downloaded without downloading
  --force-1m           Only download 1-meter data (fail if not available)
  --force-1-3-arcsec   Force 1/3 arc-second data instead of trying 1m first
  -h, --help           Show this help message

Example:
  $0 --config configs/sf_fidi.yaml --topo-dir ./topo
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
        --dry-run|--force-1m|--force-1-3-arcsec)
            EXTRA_ARGS+=("$1")
            shift
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

# Check config file exists
if [[ ! -f "$CONFIG" ]]; then
    error "Config file not found: $CONFIG"
    exit 1
fi

# Create topo directory if it doesn't exist
mkdir -p "$TOPO_DIR"

# Convert to absolute paths
CONFIG_ABS="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
TOPO_ABS="$(cd "$TOPO_DIR" && pwd)"

# Get config directory and filename
CONFIG_DIR="$(dirname "$CONFIG_ABS")"
CONFIG_FILE="$(basename "$CONFIG_ABS")"

# Always pull the latest image
info "Pulling latest container image..."
docker pull "$IMAGE" --quiet

info "Downloading DEM data..."
info "  Config: $CONFIG"
info "  Output: $TOPO_DIR"

# Run the download script inside the container
docker run --rm \
    -v "${CONFIG_DIR}:/app/configs:ro" \
    -v "${TOPO_ABS}:/app/topo" \
    --entrypoint python \
    "$IMAGE" \
    scripts/download_dem.py \
    --config "/app/configs/${CONFIG_FILE}" \
    --output-dir /app/topo \
    "${EXTRA_ARGS[@]}"

info "Done! DEM files saved to: $TOPO_DIR"

