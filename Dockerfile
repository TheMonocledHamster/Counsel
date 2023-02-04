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
RUN pip3 install --no-cache-dir -r requirements.txt

COPY model /model
COPY train.py .

USER nobody

CMD ["python3", "train.py"]
