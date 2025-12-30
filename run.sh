#!/bin/bash
# World-to-Model Runner Script
# Pulls the container from GHCR or builds locally, then runs the generator

set -e

IMAGE="ghcr.io/rdudhagra/world-to-model:latest"
LOCAL_IMAGE="world-to-model:local"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Check for Docker
if ! command -v docker &> /dev/null; then
    error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create output directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/outputs"

# Try to use the image in this order:
# 1. If GHCR image is available locally, use it
# 2. Try to pull from GHCR
# 3. Fall back to building locally
get_image() {
    # Check if GHCR image exists locally
    if docker image inspect "$IMAGE" &>/dev/null; then
        info "Using cached GHCR image: $IMAGE"
        echo "$IMAGE"
        return
    fi

    # Try to pull from GHCR
    info "Attempting to pull from GHCR: $IMAGE"
    if docker pull "$IMAGE" 2>/dev/null; then
        info "Successfully pulled: $IMAGE"
        echo "$IMAGE"
        return
    fi

    warn "Could not pull from GHCR, building locally..."

    # Build locally
    if docker build -t "$LOCAL_IMAGE" "$SCRIPT_DIR"; then
        info "Successfully built local image: $LOCAL_IMAGE"
        echo "$LOCAL_IMAGE"
        return
    else
        error "Failed to build local image"
        exit 1
    fi
}

# Show usage if no arguments
if [ $# -eq 0 ]; then
    echo "World-to-Model Terrain Generator"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Examples:"
    echo "  $0 --config configs/sf_fidi.yaml"
    echo "  $0 --config configs/bay_area.yaml --job sf_downtown"
    echo "  $0 --help"
    echo ""
    echo "The script will:"
    echo "  1. Pull the container from ghcr.io (if available)"
    echo "  2. Or build locally from Dockerfile"
    echo "  3. Mount configs/, topo/, and outputs/ directories"
    echo "  4. Run the generator with your arguments"
    echo ""

    # Still get the image and show help
    FINAL_IMAGE=$(get_image)
    docker run --rm "$FINAL_IMAGE" --help
    exit 0
fi

# Get the image to use
FINAL_IMAGE=$(get_image)

info "Running: docker run with image $FINAL_IMAGE"

# Run the container with mounts
docker run --rm \
    -v "${SCRIPT_DIR}/configs:/app/configs:ro" \
    -v "${SCRIPT_DIR}/topo:/app/topo:ro" \
    -v "${SCRIPT_DIR}/outputs:/app/outputs" \
    "$FINAL_IMAGE" "$@"

info "Done! Check the outputs/ directory for results."


