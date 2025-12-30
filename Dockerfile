# Topo-tessellate Terrain Generator
# Multi-stage build using mambaforge for meshlib compatibility

# Stage 1: Build environment
FROM condaforge/mambaforge:latest AS builder

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create the conda environment
RUN mamba env create -f /tmp/environment.yml && \
    mamba clean -afy && \
    find /opt/conda -follow -type f -name '*.a' -delete && \
    find /opt/conda -follow -type f -name '*.pyc' -delete && \
    find /opt/conda -follow -type f -name '*.js.map' -delete

# Stage 2: Runtime
FROM condaforge/mambaforge:latest

# Copy conda environment from builder
COPY --from=builder /opt/conda/envs/topo-tessellate /opt/conda/envs/topo-tessellate

# Set up environment activation
ENV PATH="/opt/conda/envs/topo-tessellate/bin:$PATH"
ENV CONDA_DEFAULT_ENV=topo-tessellate
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy application code
COPY terrain_generator/ /app/terrain_generator/
COPY configs/ /app/configs/
COPY generate.py /app/
COPY joint_cutout.stl /app/
COPY cleat_cutout.stl /app/

# Create directories for mounted volumes
RUN mkdir -p /app/outputs /app/topo

# Set the entrypoint
ENTRYPOINT ["python", "generate.py"]

# Default command shows help
CMD ["--help"]


