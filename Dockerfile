# tcrconsensus Docker image
# Build: docker build -t tcrconsensus .
# Run:   docker run -it --rm -v $(pwd)/data:/data tcrconsensus run /data/input.tsv -o /data/output

FROM python:3.10-slim

LABEL org.opencontainers.image.title="tcrconsensus"
LABEL org.opencontainers.image.description="TCR Consensus Clustering Framework"
LABEL org.opencontainers.image.version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install optional clusterers
RUN pip install --no-cache-dir \
    clustcr \
    pygliph \
    tcrdist3 \
    faiss-cpu \
    biopython

# DeepTCR is heavy (TensorFlow) — install only if needed
# RUN pip install --no-cache-dir DeepTCR

# Copy package
COPY . /app/tcrconsensus
WORKDIR /app/tcrconsensus

# Install tcrconsensus
RUN pip install -e .

# Entry point
ENTRYPOINT ["tcrconsensus"]
CMD ["--help"]
