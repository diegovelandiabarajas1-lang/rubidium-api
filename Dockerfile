FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt maturin

COPY . .

# Build Rust extension if rubidium-core exists
RUN if [ -d "rubidium-core" ]; then cd rubidium-core && maturin build --release && pip install target/wheels/*.whl; fi

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
