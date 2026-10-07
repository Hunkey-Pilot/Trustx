import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.admin import router as admin_router
from app.api.reviews import router as reviews_router
from app.api.transactions import router as transactions_router
from app.api.risk import router as risk_router
from app.db.connection import get_database_url
from app.security import require_admin
from app.services.model_loader import ModelLoader
from app.services.model_service import ModelService
from app.services.risk_engine import RiskEngine
from app.services.shap_service import ShapService
from app.services.evidence_engine import EvidenceEngine
from app.services.behavioral_engine import BehavioralEngine
from app.services.network_engine import NetworkEngine
from app.services.counterfactual_engine import CounterfactualEngine


logger = logging.getLogger("trustx")

# Largest accepted request body. A full 500-row history payload is well below this.
MAX_REQUEST_BODY_BYTES = 256 * 1024

model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.database_url = get_database_url()
    loaded_models = model_loader.load()
    app.state.model_loader = model_loader
    app.state.model_service = ModelService(loaded_models)
    app.state.risk_engine = RiskEngine()
    app.state.shap_service = ShapService(loaded_models)
    app.state.evidence_engine = EvidenceEngine()
    app.state.behavioral_engine = BehavioralEngine()
    app.state.network_engine = NetworkEngine()
    app.state.counterfactual_engine = CounterfactualEngine()
    yield


app = FastAPI(title="TrustX API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Content-Type", "X-API-Key", "X-TrustX-User"],
)


@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            too_large = int(content_length) > MAX_REQUEST_BODY_BYTES
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})
        if too_large:
            return JSONResponse(status_code=413, content={"detail": "Request body is too large"})
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    if any(error.get("type") == "json_invalid" for error in errors):
        return JSONResponse(status_code=400, content={"detail": "Malformed JSON payload"})
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": [
                {
                    "loc": [str(part) for part in error.get("loc", [])],
                    "message": str(error.get("msg", "invalid value")),
                    "type": str(error.get("type", "value_error")),
                }
                for error in errors
            ],
        },
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    # Details go to the server log only; the client receives a generic message.
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(transactions_router)
app.include_router(reviews_router)
app.include_router(risk_router)
app.include_router(admin_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "message": "TrustX API is running"}


@app.get("/api/v1/models/status", dependencies=[Depends(require_admin)])
def model_status(request: Request) -> dict:
    return request.app.state.model_loader.status()
