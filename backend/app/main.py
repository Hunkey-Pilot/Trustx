from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api.transactions import router as transactions_router
from app.api.risk import router as risk_router
from app.db.connection import get_database_url
from app.services.model_loader import ModelLoader
from app.services.model_service import ModelService
from app.services.risk_engine import RiskEngine
from app.services.shap_service import ShapService
from app.services.evidence_engine import EvidenceEngine


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
    yield


app = FastAPI(title="TrustX API", lifespan=lifespan)
app.include_router(transactions_router)
app.include_router(risk_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "message": "TrustX API is running"}


@app.get("/api/v1/models/status")
def model_status(request: Request) -> dict:
    return request.app.state.model_loader.status()
