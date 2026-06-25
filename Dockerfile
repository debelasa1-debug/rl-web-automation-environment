# ── RL Web Automation Environment ─────────────────────────────────────────
# Multi-stage Dockerfile for reproducible, isolated execution.
#
# Build:  docker build -t rl-web-env .
# Run:    docker run --rm -e ANTHROPIC_API_KEY=$KEY rl-web-env \
#           python main.py --agent rule --episodes 5
# ──────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System dependencies for Playwright Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl ca-certificates \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libasound2 \
    libpango-1.0-0 libpangocairo-1.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and its browser binaries
RUN playwright install chromium
RUN playwright install-deps chromium

# ── Copy application code ──────────────────────────────────────────────────
COPY . .

# Create output directories
RUN mkdir -p logs logs/screenshots

# ── Default command ────────────────────────────────────────────────────────
# Override with docker run ... python main.py --agent llm --episodes 10
CMD ["python", "main.py", "--agent", "rule", "--episodes", "5", "--log-level", "INFO"]
