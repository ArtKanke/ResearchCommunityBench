FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends procps curl git build-essential xxd \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy scipy scikit-learn pandas lightgbm sympy networkx fastapi "uvicorn[standard]" "pydantic>=2.10" "httpx>=0.28" litellm "anthropic==2.8.0" pyarrow
RUN useradd -m -s /bin/bash agent
COPY republic /opt/republic
COPY data /lab/data
RUN mkdir -p /lab/store /lab/logs /lab/cursors /work \
    && chown -R agent:agent /work /lab/logs /lab/cursors && chmod 700 /lab/store \
    && ln -sf /opt/republic/agent/lab /usr/local/bin/lab && chmod 755 /opt/republic/agent/lab
ENV PYTHONPATH=/opt/republic LAB_DIR=/lab WORK_DIR=/work
WORKDIR /work
