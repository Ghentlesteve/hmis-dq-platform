# HMIS Data Quality & Early-Warning Platform

End-to-end pipeline that pulls routine health data from DHIS2, scores it against the
WHO Data Quality Review framework, detects anomalies, forecasts key indicators and
shows the results on an interactive map dashboard.

> Work in progress — built in stages. See the roadmap below.

## Roadmap

1. [ ] Foundation: project layout, config, DHIS2 extraction client
2. [ ] Data lake: MinIO (S3-compatible) bronze layer
3. [ ] PySpark transforms: silver and gold tables
4. [ ] Data quality engine: WHO DQR metrics at scale
5. [ ] ML: anomaly detection and forecasting vs. baseline
6. [ ] Dashboard: choropleth map and drill-down
7. [ ] NiFi ingestion flow
8. [ ] Kubernetes deployment and CI

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
copy .env.example .env          # then fill in the values
pytest
```

## Data

Development uses the public DHIS2 demo database (synthetic data modelled on
Sierra Leone). No real patient or national HMIS data is stored in this repository.
