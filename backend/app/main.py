from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.transactions import router as transactions_router
from app.api.risk import router as risk_router
from app.db.connection import get_database_url
from app.services.model_loader import ModelLoader
from app.services.model_service import ModelService
from app.services.risk_engine import RiskEngine
from app.services.shap_service import ShapService
from app.services.evidence_engine import EvidenceEngine
from app.services.behavioral_engine import BehavioralEngine
from app.services.network_engine import NetworkEngine
from app.services.counterfactual_engine import CounterfactualEngine


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
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(transactions_router)
app.include_router(risk_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "message": "TrustX API is running"}


@app.get("/api/v1/models/status")
def model_status(request: Request) -> dict:
    return request.app.state.model_loader.status()
