FROM python:3.10.8

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY source /source

# Set working directory
WORKDIR /source

# Run the application
CMD ["python3", "main.py"]
