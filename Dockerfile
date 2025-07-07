FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Copy all necessary files
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY helper.py helper.py
COPY settings.py settings.py
COPY weights/best(trainedOnM6).pt weights/best(trainedOnM6).pt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create .streamlit directory and config file to prevent crash
RUN mkdir -p /app/.streamlit && \
    echo "\
[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
" > /app/.streamlit/config.toml

# Expose the Streamlit port
EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
