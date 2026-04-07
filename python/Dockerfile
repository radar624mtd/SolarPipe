# SolarPipe Python gRPC sidecar — Docker image.
#
# RULE-060: Lifecycle managed by .NET IHostedService (SidecarLifecycleService).
# RULE-062: Server binds to 0.0.0.0:50051 (not localhost — unreachable from host otherwise).
#
# Usage:
#   docker build -f python/Dockerfile -t solarpipe-sidecar:latest .
#   docker run -p 50051:50051 --rm solarpipe-sidecar:latest
#
# GPU variant (CUDA 12.1):
#   FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS base
#   (swap the base image below; everything else is the same)

FROM python:3.12-slim AS base

# Prevent .pyc files and enable stdout logging (ADR-010: JSON to stdout in containers)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for pyarrow native extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies — pinned versions from requirements.txt
COPY python/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy proto-generated stubs and server
COPY python/solarpipe_pb2.py      solarpipe_pb2.py
COPY python/solarpipe_pb2_grpc.py solarpipe_pb2_grpc.py
COPY python/solarpipe_server.py   solarpipe_server.py

# Logs directory (structured JSON, ADR-010)
RUN mkdir -p /app/logs

EXPOSE 50051

# RULE-062: bind to 0.0.0.0:50051 (--port arg forwarded to server.add_insecure_port)
ENTRYPOINT ["python", "solarpipe_server.py", "--port", "50051"]
