# 🤖 PriceAI — ML Dynamic Pricing Microservice

A **FastAPI** microservice that predicts dynamic room pricing for each date based on occupancy and demand factors. Called by the ASP.NET Core backend to populate the booking calendar with real-time prices.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [API Endpoints](#api-endpoints)
- [Input Features](#input-features)
- [Installation & Launch](#installation--launch)
- [Performance Benchmark](#performance-benchmark)

---

## Overview

PriceAI implements a **dynamic pricing algorithm** for hotel rooms.  
The model takes the following factors into account:

| Factor | Description |
|---|---|
| `lead_time` | Number of days before check-in |
| `occupancy_rate` | Current hotel occupancy rate (0.0–1.0) |
| `is_weekend` | Weekend flag (0 / 1) |
| `pickup_rate` | Booking speed over the last 7 days |
| `base_price` | Base room price per night |

The output is an adjusted price returned as a `{ "YYYY-MM-DD": price }` dictionary and displayed in the interactive calendar on the frontend.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  PriceAI (FastAPI)                  │
│  ┌──────────────┐   ┌──────────────┐               │
│  │  /predict-   │   │  /predict-   │               │
│  │   price      │   │  prices-     │               │
│  │  (single)    │   │  batch       │               │
│  └──────┬───────┘   └──────┬───────┘               │
│         │                  │                        │
│         └────────┬─────────┘                        │
│                  ▼                                  │
│         [ ML Model / Rule Engine ]                  │
│                  ▼                                  │
│         [ Adjusted Price ]                          │
└─────────────────────────────────────────────────────┘
         ▲ called from
┌────────────────────┐
│  ASP.NET Core API  │  GET /api/Rooms/Free
│                    │  GET /api/Rooms/{id}/daily-prices
└────────────────────┘
```

---

## API Endpoints

### `GET /health`
Health check — verifies the service is running and the model is loaded.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### `POST /predict-price`
Predicts the price for a **single** set of input parameters.

**Request body:**
```json
{
  "lead_time": 14,
  "occupancy_rate": 0.6,
  "is_weekend": 0,
  "pickup_rate": 5.0,
  "base_price": 5000.0
}
```

**Response:**
```json
{
  "predicted_price": 5430.0
}
```

---

### `POST /predict-prices-batch`
Batch prediction for a **date range** — used to fill the booking calendar.

**Request body:**
```json
{
  "items": [
    {
      "date": "2026-08-01",
      "lead_time": 30,
      "occupancy_rate": 0.5,
      "is_weekend": 0,
      "pickup_rate": 3.0,
      "base_price": 5000.0
    },
    {
      "date": "2026-08-02",
      "lead_time": 29,
      "occupancy_rate": 0.5,
      "is_weekend": 0,
      "pickup_rate": 3.0,
      "base_price": 5000.0
    }
  ]
}
```

**Response:**
```json
{
  "2026-08-01": 4900.0,
  "2026-08-02": 4850.0
}
```

---

## Input Features

| Field | Type | Range | Description |
|---|---|---|---|
| `lead_time` | `int` | 0–365 | Days until check-in |
| `occupancy_rate` | `float` | 0.0–1.0 | Hotel occupancy |
| `is_weekend` | `int` | 0 or 1 | Weekend flag |
| `pickup_rate` | `float` | 0.0–∞ | Booking pace |
| `base_price` | `float` | > 0 | Base room price |
| `date` | `string` | `YYYY-MM-DD` | Batch requests only |

---

## Installation & Launch

### Requirements

- Python 3.10+
- pip

### Install dependencies

```bash
pip install fastapi uvicorn scikit-learn numpy pandas
```

### Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Service available at: `http://localhost:8000`

Interactive Swagger UI: `http://localhost:8000/docs`

---

## Performance Benchmark

Use the `benchmark.py` script in the project root to measure performance.

**Make sure both services are running before starting:**
- Backend (ASP.NET Core): `http://localhost:5237`
- PriceAI (FastAPI): `http://localhost:8000`

```bash
python benchmark.py
```

**Sample output:**

```
┌───────────────────────────────────────────────┬──────────┬──────────┬──────────┬──────────┬────────┐
│ Endpoint                                      │  Avg     │  Median  │   Min    │   Max    │ Errors │
│                                               │  (ms)    │  (ms)    │   (ms)   │   (ms)   │        │
├───────────────────────────────────────────────┼──────────┼──────────┼──────────┼──────────┼────────┤
│ GET /health                                   │     12.3 │     11.9 │      9.8 │     18.1 │      0 │
│ POST /predict-price (single)                  │     18.5 │     17.2 │     14.0 │     28.3 │      0 │
│ POST /predict-prices-batch (30 dates)         │     25.1 │     24.0 │     19.5 │     36.7 │      0 │
│ POST /predict-prices-batch (60 dates)         │     38.4 │     37.2 │     29.1 │     55.0 │      0 │
└───────────────────────────────────────────────┴──────────┴──────────┴──────────┴──────────┴────────┘
```

**Performance target:** average response time < 500 ms.
