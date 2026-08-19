from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from app.core.config import settings
from app.schemas.question import QuestionPairRequest, PredictionResponse
from app.services.predictor import get_predictor

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return request.app.state.templates.TemplateResponse(
        "index.html",
        {"request": request, "app_name": settings.app_name}
    )

@router.get("/health")
async def health():
    try:
        predictor = get_predictor()
        return {
            "status": "healthy",
            "model_loaded": predictor.model is not None,
            "version": settings.app_version,
        }
    except Exception:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "version": settings.app_version,
        }

@router.get("/api/v1/info")
async def model_info():
    predictor = get_predictor()
    return predictor.metadata

@router.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(payload: QuestionPairRequest):
    try:
        return get_predictor().predict(payload.question1, payload.question2)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs for details."
        ) from exc
