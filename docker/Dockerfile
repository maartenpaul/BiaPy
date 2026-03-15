FROM ghcr.io/prefix-dev/pixi:0.49.0-noble-cuda-12.8.1

LABEL maintainer="Daniel Franco-Barranco <daniel.franco@dipc.org>"
LABEL description="BiaPy: Accessible deep learning on bioimages (pixi + CUDA)"

USER root
WORKDIR /root

# System libraries needed by OpenCV and headless rendering
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libfontconfig1 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PIXI_CACHE_DIR=/root/.pixi/cache
RUN mkdir -p ${PIXI_CACHE_DIR}

# Copy BiaPy source first (the pypi-dependency in pixi.toml points to /src)
COPY . /src

# Copy pixi manifest into the working directory
COPY docker/pixi.toml /root/pixi.toml

# Resolve and install all dependencies (conda + pip install of biapy from /src)
RUN pixi install --manifest-path /root/pixi.toml

# Create the shell-hook script so the pixi env is activated in all shells
RUN pixi shell-hook --manifest-path /root/pixi.toml > /shell-hook.sh && \
    chmod +x /shell-hook.sh
RUN pixi shell-hook --manifest-path /root/pixi.toml >> ~/.bashrc

# Smoke tests
RUN /bin/bash -c "source /shell-hook.sh && python -c 'from biapy import predict; print(\"BiaPy predict API: OK\")'"
RUN /bin/bash -c "source /shell-hook.sh && biapy --version"
RUN /bin/bash -c "source /shell-hook.sh && biapy predict --help"

# Entrypoint activates the pixi environment, then runs the user command.
# This makes the container work seamlessly with nextflow:
#   docker run biapy:pixi biapy predict --model m.pth --input /data/images/
ENTRYPOINT ["/bin/bash", "-c", "source /shell-hook.sh && exec \"$@\"", "--"]
CMD ["biapy", "--version"]
