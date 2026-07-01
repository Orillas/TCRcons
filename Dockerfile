# tcrconsensus Docker image
# Build: docker build -t tcrconsensus .
# Run:   docker run -it --rm -v $(pwd)/data:/data tcrconsensus run /data/input.tsv -o /data/output

FROM python:3.10-slim

LABEL org.opencontainers.image.title="tcrconsensus"
LABEL org.opencontainers.image.description="TCR Consensus Clustering Framework"
LABEL org.opencontainers.image.version="1.1.1"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy package
COPY . /app/tcrconsensus
WORKDIR /app/tcrconsensus

# Install tcrconsensus + the CPU-installable backend (tcrdist3; needs the
# build-essential toolchain already installed above for the parasail C lib).
RUN pip install -e ".[tcrdist3]"

# DeepTCR (TensorFlow) — two-step workflow for reproducible pinned deps.
# Uncomment on a CUDA-capable host:
#   RUN pip install --no-deps "DeepTCR @ git+https://github.com/sidhomj/DeepTCR.git@3930ca05a987c7cc621b4f2ecfd740e2d62799d8"
#   COPY requirements/deeptcr-pinned.txt /tmp/deeptcr-pinned.txt
#   RUN pip install -r /tmp/deeptcr-pinned.txt
# clusTCR is NOT on PyPI and pins scipy==1.8 (conflicts with scipy>=1.9); add
# manually if needed:
#   RUN pip install --no-deps "clustcr @ git+https://github.com/svalkiers/clusTCR.git"
#   RUN pip install markov-clustering faiss-cpu==1.7.4
# GLIPH2 / GIANA / TCRMatch are external binaries — configure via TCR_* env vars.

# Entry point
ENTRYPOINT ["tcrconsensus"]
CMD ["--help"]
