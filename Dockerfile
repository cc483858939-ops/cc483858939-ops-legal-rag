FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ARG INSTALL_MODEL_EXTRAS=false
ARG INSTALL_DEV=false

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY requirements-dev.txt ./
COPY requirements-models.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_DEV" = "true" ]; then pip install --no-cache-dir -r requirements-dev.txt; fi \
    && if [ "$INSTALL_MODEL_EXTRAS" = "true" ]; then pip install --no-cache-dir -r requirements-models.txt; fi

COPY src ./src
RUN pip install --no-cache-dir --no-build-isolation --no-deps -e .

COPY configs ./configs
COPY data ./data
COPY scripts ./scripts
COPY tests ./tests

CMD ["personal-notes-rag", "--help"]
