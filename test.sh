#!/bin/bash
# Topo-tessellate Test Runner
# Runs pytest inside the Docker container

set -e

IMAGE="ghcr.io/rdudhagra/topo-tessellate:latest"
LOCAL_IMAGE="topo-tessellate:local"
TEST_IMAGE="topo-tessellate:test"

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

# Build the test image
build_image() {
    info "Building test image..."
    docker build -t "$TEST_IMAGE" "$SCRIPT_DIR"
}

# Check if we have a usable image
get_image() {
    # First check if test image exists
    if docker image inspect "$TEST_IMAGE" &>/dev/null; then
        info "Using existing test image: $TEST_IMAGE"
        echo "$TEST_IMAGE"
        return
    fi

    # Check if local image exists
    if docker image inspect "$LOCAL_IMAGE" &>/dev/null; then
        info "Using local image: $LOCAL_IMAGE"
        echo "$LOCAL_IMAGE"
        return
    fi

    # Try to pull from GHCR
    info "Attempting to pull from GHCR: $IMAGE"
    if docker pull "$IMAGE" 2>/dev/null; then
        info "Successfully pulled: $IMAGE"
        echo "$IMAGE"
        return
    fi

    # Build locally as fallback
    warn "No image found, building locally..."
    build_image
    echo "$TEST_IMAGE"
}

# Parse arguments
REBUILD=false
PYTEST_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild|-r)
            REBUILD=true
            shift
            ;;
        --help|-h)
            echo "Topo-tessellate Test Runner"
            echo ""
            echo "Usage: $0 [OPTIONS] [PYTEST_ARGS]"
            echo ""
            echo "Options:"
            echo "  --rebuild, -r    Force rebuild of the test image"
            echo "  --help, -h       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run all tests"
            echo "  $0 -v                        # Run with verbose output"
            echo "  $0 tests/test_buildings_processor.py  # Run specific test file"
            echo "  $0 -k 'bbox'                 # Run tests matching 'bbox'"
            echo "  $0 --rebuild                 # Rebuild image and run tests"
            exit 0
            ;;
        *)
            PYTEST_ARGS="$PYTEST_ARGS $1"
            shift
            ;;
    esac
done

# Rebuild if requested
if [ "$REBUILD" = true ]; then
    info "Forcing rebuild..."
    build_image
    FINAL_IMAGE="$TEST_IMAGE"
else
    FINAL_IMAGE=$(get_image)
fi

info "Running tests in container: $FINAL_IMAGE"

# Run pytest in container
docker run --rm \
    --entrypoint pytest \
    "$FINAL_IMAGE" \
    $PYTEST_ARGS

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    info "✅ All tests passed!"
else
    error "❌ Tests failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE

