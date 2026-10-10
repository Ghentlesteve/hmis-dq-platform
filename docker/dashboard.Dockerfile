# Dashboard image: Python + Streamlit + DuckDB + this project. No Java or Spark:
# the dashboard only reads the lake's Parquet files (about 1 GB, against 4 GB for
# the Spark image; most of it is pyarrow, pandas and numpy, which Streamlit needs).
#
#   docker compose up -d                       (builds it the first time)
#   docker compose up -d --build dashboard     (after code changes)
#   open http://localhost:8501

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Streamlit settings for a server: listen on all interfaces inside the
    # container, don't try to open a browser, don't send usage statistics
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN useradd --create-home --uid 1001 hmis
WORKDIR /app

# 1. Dependencies only, from pyproject.toml with an empty package (and README) in
#    place of the real ones: this layer is rebuilt only when pyproject.toml
#    changes, and the pip cache mount keeps downloads between builds. hatchling
#    (the build backend) is installed too, so step 3 needn't download it.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p src/hmis_dq && touch src/hmis_dq/__init__.py README.md \
    && pip install hatchling ".[dashboard]" \
    && pip uninstall -y hmis-dq \
    && rm -rf src README.md

# 2. DuckDB downloads its S3 extension (httpfs) on first use, into the user's home.
#    Fetch it now, as that user, so the container starts without the internet.
USER hmis
RUN python -c "import duckdb; duckdb.sql('INSTALL httpfs')"
USER root

# 3. The code (seconds to rebuild)
COPY README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps --no-build-isolation .
COPY .streamlit ./.streamlit
USER hmis

EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"
CMD ["hmis-dq", "dashboard", "--port", "8501"]
