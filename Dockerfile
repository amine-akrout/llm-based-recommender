FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama

# RUN curl -fsSL https://ollama.com/install.sh | sh

# # Pull the LLaMA 3.2 (3B) model inside the container
# RUN ollama run llama3.2:3b

# Serve Ollama

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .


# Run the application.
CMD ["uvicorn", "app.src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
