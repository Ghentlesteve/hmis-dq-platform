# HMIS Data Quality & Early-Warning Platform

End-to-end pipeline that pulls routine health data from DHIS2, scores it against the
WHO Data Quality Review framework, detects anomalies, forecasts key indicators and
shows the results on an interactive map dashboard.

> Work in progress — built in stages. See the roadmap below.

## Roadmap

1. [x] Foundation: project layout, config, DHIS2 extraction client
2. [x] Data lake: S3-compatible object store (SeaweedFS), bronze layer
3. [x] PySpark transforms: silver and gold tables
4. [x] Data quality engine: WHO DQR metrics at scale
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
docker compose run --rm spark hmis-dq spark build   # bronze -> silver -> gold -> DQ
docker compose run --rm spark-test                  # Spark unit tests (Linux)
```

## Data quality engine

Every check writes to one `findings` table (check, WHO DQR dimension, severity,
district / facility / indicator / period, value, expected value, score, and a
plain-language message). Thresholds live in one place
([`rules.py`](src/hmis_dq/spark/dq/rules.py)) with their source.

| WHO DQR dimension | Checks |
|---|---|
| Completeness | facilities that never reported; facilities below the 80% benchmark |
| Outliers | 3 SD from the facility's own mean, and modified z-score (median/MAD) >= 3.5 |
| Internal consistency | negative Penta1->Penta3 and ANC1->ANC4 drop-out per year; OPV1/Penta1 ratio vs national |
| Consistency over time | district totals vs the mean of up to 3 previous years, same months, +/-33% |
| System | values identical to last year's; same value months in a row; impossible timestamps; reports entered before the month ended; datasets assigned to non-facilities; facilities without coordinates |

Each facility and district gets a 0-100 score per dimension (completeness,
accuracy, consistency, integrity) and an overall grade. Every score is
`100 x (1 - share affected)`, so it can be explained from counts, and a dimension
that can't be measured is left out rather than counted as perfect.

## Data quality findings (DHIS2 demo database)

Run over 1,021,137 values from 1,169 facilities, January 2023 to September 2026.

| | Child Health | Reproductive Health |
|---|---|---|
| Reporting completeness | **18.0%** (9,357 of 51,885 reports) | **84.1%** (20,383 of 24,234) |
| Timeliness | unknown (see below) | unknown |
| Overall score | **48.8 (D)** | **80.3 (B)** |

1. **Copied, not counted.** 69-92% of each facility's monthly values are identical
   to the same month the year before (1,748 facility-years flagged). This is also
   why *consistency over time* finds nothing: totals are stable because they are
   copies. A check that looks perfect on its own is explained by another.
2. **Four in five facilities don't report Child Health.** 1,161 facilities are
   assigned the dataset; only 237 ever report it. District totals rest on
   fewer than a fifth of facilities.
3. **Timeliness can't be measured.** Every one of the 29,740 received reports has an
   entry date before its month ended (up to 16 years early), and 40% of values were
   "last updated" before they were created. Rather than report a misleading 100%
   on time, timeliness is marked unknown.
4. **Facility errors hidden by district totals.** 366 facility-years report more
   Penta3 than Penta1 doses, or more ANC4 than ANC1 visits (negative drop-out);
   no district total shows it.
5. **Configuration issues.** 5 dataset assignments point at districts or the whole
   country instead of facilities, and 48% of facilities have no GPS coordinates.

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
