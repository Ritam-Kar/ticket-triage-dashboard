FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

ENTRYPOINT ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
