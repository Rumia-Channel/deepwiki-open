# syntax=docker/dockerfile:1-labs

# Build argument for custom certificates directory
ARG CUSTOM_CERT_DIR="certs"

FROM node:20-alpine3.22 AS node_base

FROM node_base AS node_deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --legacy-peer-deps

FROM node_base AS node_builder
WORKDIR /app
COPY --from=node_deps /app/node_modules ./node_modules
# Copy only necessary files for Next.js build
COPY package.json package-lock.json next.config.ts tsconfig.json tailwind.config.js postcss.config.mjs ./
COPY src/ ./src/
COPY public/ ./public/
# Increase Node.js memory limit for build and disable telemetry
ENV NODE_OPTIONS="--max-old-space-size=4096"
ENV NEXT_TELEMETRY_DISABLED=1
RUN NODE_ENV=production npm run build

FROM python:3.11-slim AS py_deps
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv --no-cache-dir && \
    uv sync --frozen --no-dev --no-editable

# Use Python 3.11 as final image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Node.js and npm
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    git \
    ca-certificates \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key -o /tmp/nodesource-key.gpg \
    && gpg --show-keys --with-fingerprint /tmp/nodesource-key.gpg 2>&1 | grep -q "9FD3 B784 BC1C 6FC3 1A8A  1A86 FC51 3319 1624 779A" || \
       (echo "WARNING: NodeSource GPG key fingerprint mismatch! Proceeding anyway..." >&2) \
    && cat /tmp/nodesource-key.gpg | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && rm /tmp/nodesource-key.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Update certificates if custom ones were provided and copied successfully
RUN if [ -n "${CUSTOM_CERT_DIR}" ]; then \
        mkdir -p /usr/local/share/ca-certificates && \
        if [ -d "${CUSTOM_CERT_DIR}" ]; then \
            cp -r ${CUSTOM_CERT_DIR}/* /usr/local/share/ca-certificates/ 2>/dev/null || true; \
            update-ca-certificates; \
            echo "Custom certificates installed successfully."; \
        else \
            echo "Warning: ${CUSTOM_CERT_DIR} not found. Skipping certificate installation."; \
        fi \
    fi

ENV PATH="/opt/venv/bin:$PATH"

# Copy Python dependencies
COPY --from=py_deps /app/.venv /opt/venv
COPY api/ ./api/

# Copy Node app
COPY --from=node_builder /app/public ./public
COPY --from=node_builder /app/.next/standalone ./
COPY --from=node_builder /app/.next/static ./.next/static

# Expose the port the app runs on
EXPOSE ${PORT:-8001} 3000

# Create a script to run both backend and frontend
RUN echo '#!/bin/bash\n\
set -euo pipefail\n\
\n\
# Check for required environment variables\n\
if [ -z "${OPENAI_API_KEY:-}" ] || [ -z "${GOOGLE_API_KEY:-}" ]; then\n\
  echo "Warning: OPENAI_API_KEY and/or GOOGLE_API_KEY environment variables are not set."\n\
  echo "These are required for DeepWiki to function properly."\n\
  echo "You can provide them via a mounted .env file or as environment variables when running the container."\n\
fi\n\
\n\
# Start the API server in the background with the configured port\n\
python -m api.main &\n\
PORT=3000 HOSTNAME=0.0.0.0 node server.js &\n\
wait -n\n\
exit $?' > /app/start.sh && chmod +x /app/start.sh

# Create non-root user and fix ownership
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app/api/logs && \
    chown -R appuser:appuser /app /opt/venv && \
    chown appuser:appuser /root/.adalflow 2>/dev/null || true

USER appuser

# Set environment variables
ENV PORT=8001
ENV NODE_ENV=production
ENV SERVER_BASE_URL=http://localhost:${PORT:-8001}

# Create empty .env file (will be overridden if one exists at runtime)
RUN touch .env

# Command to run the application
CMD ["/app/start.sh"]
