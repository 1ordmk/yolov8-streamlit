FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY app.py app.py
COPY helper.py helper.py
COPY settings.py settings.py
COPY weights/best(trainedOnM6).pt weights/best(trainedOnM6).pt

RUN pip install --no-cache-dir -r requirements.txt

# ✅ Create streamlit config directory inside /app
RUN mkdir -p /app/.streamlit

# ✅ Set config so it doesn't default to root
ENV STREAMLIT_CONFIG_DIR=/app/.streamlit
ENV STREAMLIT_HOME=/app/.streamlit
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false

# ✅ Add Streamlit config file
RUN echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = 8501\n\
" > /app/.streamlit/config.toml

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]

