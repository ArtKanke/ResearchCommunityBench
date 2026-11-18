FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends procps curl git xxd && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade torchvision==0.20.1 numpy scipy scikit-learn pandas "brotli>=1.1" fastapi "uvicorn[standard]" "pydantic>=2.10" "httpx>=0.28" litellm "anthropic==2.8.0" pyarrow
COPY prepare_cifar.py /tmp/prepare_cifar.py
RUN python /tmp/prepare_cifar.py /lab/data/cifar10 && rm /tmp/prepare_cifar.py
RUN useradd -m -s /bin/bash agent
COPY republic /opt/republic
COPY data /lab/data
RUN mkdir -p /lab/store /lab/logs /lab/cursors /work \
    && chown -R agent:agent /work /lab/logs /lab/cursors && chmod 700 /lab/store \
    && ln -sf /opt/republic/agent/lab /usr/local/bin/lab && chmod 755 /opt/republic/agent/lab
ENV PYTHONPATH=/opt/republic LAB_DIR=/lab WORK_DIR=/work LAB_GPU_SLOTS=1
WORKDIR /work
