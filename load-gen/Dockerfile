FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip && \
    mkdir /load

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# Copy source code
COPY arrival_rates /load/arrival_rates
COPY loadserver.py /load/loadserver.py

# Set user
USER nobody

# Expose uvicorn port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "load.loadserver:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"]
