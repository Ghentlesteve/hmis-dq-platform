# HMIS Data Quality & Early-Warning Platform

End-to-end pipeline that pulls routine health data from DHIS2, scores it against the
WHO Data Quality Review framework, detects anomalies, forecasts key indicators and
shows the results on an interactive map dashboard.

> Work in progress — built in stages. See the roadmap below.

## Roadmap

1. [x] Foundation: project layout, config, DHIS2 extraction client
2. [x] Data lake: S3-compatible object store (SeaweedFS), bronze layer
3. [x] PySpark transforms: silver and gold tables
4. [ ] Data quality engine: WHO DQR metrics at scale
5. [ ] ML: anomaly detection and forecasting vs. baseline
6. [ ] Dashboard: choropleth map and drill-down
7. [ ] NiFi ingestion flow
8. [ ] Kubernetes deployment and CI

## Local setup

Requires Python 3.11+ and Docker Desktop.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
copy .env.example .env          # then fill in the values
docker compose up -d            # start the local S3 data lake
pytest                          # unit tests + lake integration test
```

## Usage

```bash
hmis-dq ping                    # check the DHIS2 connection
hmis-dq extract --start 2023-01 # DHIS2 -> bronze bucket (resumable)
hmis-dq lake status             # what's in the lake
hmis-dq lake upload             # copy a local data/raw folder into the lake
```

Spark jobs run in a container next to the lake:

```bash
docker compose build spark spark-test
docker compose run --rm spark hmis-dq spark build   # bronze -> silver -> gold
docker compose run --rm spark-test                  # Spark unit tests (Linux)
```

## Lake layout

| Layer | Bucket | Contents |
|---|---|---|
| Bronze | `bronze` | DHIS2 responses exactly as received (gzipped JSON) plus lineage metadata |
| Silver | `silver` | Typed Parquet: `data_values` (with a parse status per value), org units with hierarchy and coordinates, data elements, datasets, dataset assignments |
| Gold | `gold` | `facility_month`, `reporting` (one row per *expected* report, received or not), `district_month` with completeness |

Extraction pulls one chunk per dataset x district x month, in parallel. Re-runs
skip chunks already stored and re-fetch only the last few months, where late
reports still arrive.

## Data

Development uses the public DHIS2 demo database (synthetic data modelled on
Sierra Leone). No real patient or national HMIS data is stored in this repository.
