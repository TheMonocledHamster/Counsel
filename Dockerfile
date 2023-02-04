FROM ecpe4s/ubuntu22.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y mpich && \
    apt-get install -y --no-install-recommends python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install -U pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    rm requirements.txt && \
    mkdir /counsel

WORKDIR /counsel
COPY model /counsel/model
COPY train.py           \
     model_eval.sh      \
     inference.py       \
     /counsel/

RUN chown -R nobody:nogroup /counsel
USER nobody

CMD ["bash", "model_eval.sh", "&&", \
     "python3", "inference.py"]
