FROM python:3.10-slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    openmpi-bin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    rm requirements.txt && \
    mkdir /Counsel

WORKDIR /Counsel
COPY model              \
     train.py           \
     model_eval.sh      \
     plotting           \
     inference.py       \
     ./

USER nobody

CMD ["bash", "model_eval.sh", "&&", \
     "python3", "inference.py"]
