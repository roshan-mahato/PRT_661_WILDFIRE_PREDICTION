# PRT_661_WILDFIRE_PREDICTION

## Real-Time Wildfire Risk Tracking and Prediction System for Australia

A real-time, web-based system that tracks wildfire risk and predicts fire danger up to 7 days (168 hours) ahead across Australia. The system combines satellite hotspot data, weather forecasts, and a hybrid physics + machine learning engine to support emergency services, land managers, researchers, and local communities.


---

## Project Overview

The system ingests real-time satellite fire data and numerical weather forecasts, processes them through a validated ETL pipeline, and applies both the **McArthur Forest Fire Danger Index (FFDI)** and a supervised ML classifier (XGBoost / Random Forest) to estimate fire danger ratings and rapid-spread probabilities at a local, sub-regional scale.

### Core Objectives
- **Asynchronous Data Ingestion** — Low-latency REST connections to NASA FIRMS (satellite hotspots), Open-Meteo (7-day weather forecasts), and air quality data.
- **Feature Engineering & ETL** — Spatial filtering (Haversine radial buffer), AEST timestamp conversion, and fuel moisture index generation.
- **Hybrid Analytics Engine** — McArthur FFDI physics baseline combined with a tree-based ML classifier for rapid fire-spread probability.
- **Persistent Storage** — 3NF-normalised SQLite database for caching, query logs, predictions, and offline fallback.
- **Interactive Dashboard** — Streamlit app with Folium GIS mapping and Plotly forecast timelines.

---

## Theme Alignment

This project addresses **Theme 2: Predictive Analytics and Forecasting**, covering the full data science pipeline:

| Pillar | Implementation |
|---|---|
| Data Acquisition | NASA FIRMS, Open-Meteo APIs |
| Data Storage | SQLite (`wildfire_demo.db`) |
| Data Processing | Pandas, NumPy |
| Analytics & Modelling | McArthur FFDI + XGBoost/Random Forest |
| Visualization | Streamlit, Folium, Plotly |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Acquisition | Python, async REST, Open-Meteo, NASA FIRMS |
| Data Processing | Pandas, NumPy |
| Fire Analytics | McArthur FFDI (Python implementation) |
| Storage | SQLite (`wildfire_demo.db`) |
| GIS | Folium |
| Dashboard | Streamlit |
| Charts | Plotly |
| Project Management | GitHub + Jira |

---

## Repository Structure

```
├── docs/         # Reports and design documents
├── diagrams/     # Draw.io architecture and workflow exports
├── src/          # Application source code
├── data/         # Data instructions and small permitted samples
├── tests/        # Validation and test scripts
└── streamlit/    # Local configuration (no secrets committed)
```

---

## Team Members

Member ID| Name | Student ID |
|---|---|
Member 1| Roshan Mahato | S390410 |
Member 2| Sansuwa Shrestha | S395173 |
Member 3| Anish Machamasi | S389151 |
Member 4| Salin Panta | S395229 |
Member 5| Minh Nguyet Tran | S394122 |

**Submitted to:** Reem Sherif — PRT661, Charles Darwin University, Sydney Campus

---

## Project Management

- **Jira Board:** [Team Data Science Practice – Jira board](https://anishmachamasi2262.atlassian.net/jira/software/projects/KAN/boards/2?filter=&groupBy=none&atlOrigin=eyJpIjoiYzk2Y2U2MmI1ZmY2NDFmYWEyZjJiYTNlOGE3MTJhMDMiLCJwIjoiaiJ9)


---

## Timeline

| Week | Activities |
|---|---|
| Week 1 | Project setup, architecture, API exploration, Jira/GitHub setup |
| Week 2 | API integration, validation, SQLite schema, FFDI implementation |
| Week 3 | Streamlit dashboard, Folium GIS, Plotly forecast charts |
| Week 4 | Testing, risk review, documentation, report and demo prep |

---

## Ethics & Privacy

- **Zero PII collection** — only spatial coordinates and meteorological data are processed.
- **Open data compliance** with NASA FIRMS and Open-Meteo licensing terms.
- **Secret management** via `.streamlit/secrets.toml`, excluded from version control.

---

## References

Key references and full citation list are available in the project proposal report (`docs/`).
