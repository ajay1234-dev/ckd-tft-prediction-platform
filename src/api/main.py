"""
src/api/main.py
FastAPI application entry point for Phase 3 CKD Prediction & Explainability API.

Start with:
    uvicorn src.api.main:app --reload --port 8000

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.dependencies import initialize_dependencies
from src.api.routes import health, model_info, patient, predict, explain
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DESCRIPTION = """
## CKD Progression Prediction API — Phase 3

**Research prototype** built on a Temporal Fusion Transformer (TFT) trained on
*synthetically generated* longitudinal CKD data.

> **NOT FOR CLINICAL USE.** Predictions and explanations are for research purposes only.
> This system has not been clinically validated and must not be used to inform medical decisions.

### Endpoints
| Endpoint | Description |
|---|---|
| `GET /health` | Service liveness |
| `GET /model/info` | Model metadata and configuration |
| `GET /patient/{id}` | Synthetic patient visit history |
| `POST /predict` | Multi-task, multi-horizon CKD predictions |
| `POST /explain` | Predictions + TFT-native explainability |

### Explainability Methods
- **Variable Selection Networks (VSN):** Shows which features the model assigned importance to.
- **Temporal Self-Attention:** Shows which historical visits influenced the prediction most.

⚠️ These are model interpretability signals — they do **not** establish causal relationships.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    logger.info("=== CKD Prediction API starting up ===")
    try:
        initialize_dependencies()
        logger.info("=== API ready to serve requests ===")
    except FileNotFoundError as exc:
        logger.critical(f"STARTUP FAILED — missing artifact: {exc}")
        raise
    except Exception as exc:
        logger.critical(f"STARTUP FAILED — unexpected error: {exc}", exc_info=True)
        raise
    yield
    logger.info("=== CKD Prediction API shutting down ===")


app = FastAPI(
    title="CKD Progression Prediction API",
    description=_DESCRIPTION,
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(model_info.router)
app.include_router(patient.router)
app.include_router(predict.router)
app.include_router(explain.router)


# ── Global exception handler — never leak internal details ───────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Check server logs."},
    )


@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "CKD Progression Prediction API",
        "version": "3.0.0",
        "docs": "/docs",
        "disclaimer": (
            "Research prototype using synthetically generated CKD data. "
            "Not for clinical use."
        ),
    }
