FROM ecpe4s/ubuntu20.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libopenmpi-dev \
    python3.10 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install -U pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    rm requirements.txt && \
    mkdir /counsel

WORKDIR /counsel
COPY model /counsel//model
COPY train.py           \
     model_eval.sh      \
     inference.py       \
     /counsel/

# ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
#     OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

CMD ["bash", "model_eval.sh", "&&", \
     "python3", "inference.py"]
