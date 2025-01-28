FROM python:3.12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Upgrade pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools

# Copy the dependencies file to the working directory
COPY requirements.txt .
# RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run the application.
CMD ["uvicorn", "app.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

