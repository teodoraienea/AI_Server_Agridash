import base64
import io
import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

try:
    from openai import OpenAI
except ImportError:  # Keeps the AI service runnable even before `pip install openai`.
    OpenAI = None

MODEL_PATH = "models/yolo11m/best.pt"
MODEL_NAME = "YOLO11m"
OPENAI_TREATMENT_MODEL = os.getenv("OPENAI_TREATMENT_MODEL", "gpt-4o-mini")

app = FastAPI(title="AgriDash Plant Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(MODEL_PATH)


class ClassifyRequest(BaseModel):
    image: str


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "AgriDash AI service is running"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "success",
        "service": "AGRIDASH_Ai_Service",
        "model": MODEL_NAME,
        "model_path": MODEL_PATH,
        "classes": model.names,
    }


def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def humanize_label(label: str) -> str:
    return (
        label.strip()
        .replace("___", " - ")
        .replace("__", " - ")
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )


def local_treatment_fallback(label: str) -> str:
    normalized_label = normalize_label(label)
    readable_label = humanize_label(label)

    if normalized_label in {"healthy", "no_disease_detected", "no_disease"}:
        return "No disease detected. Continue regular monitoring, keep irrigation balanced, and inspect new leaves every few days."

    return (
        f"Possible {readable_label} detected. Remove the most affected leaves or plant parts, avoid overhead watering, "
        "improve air circulation, and keep the plant area clean from infected debris. Monitor the plant for the next few days; "
        "if symptoms spread, use only locally approved plant-protection products according to the label and ask a certified "
        "agronomist before applying chemical treatments."
    )


@lru_cache(maxsize=128)
def openai_treatment_for(label: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    readable_label = humanize_label(label)

    if OpenAI is None or not api_key:
        return local_treatment_fallback(label)

    client = OpenAI(api_key=api_key, timeout=8.0)

    response = client.responses.create(
        model=OPENAI_TREATMENT_MODEL,
        instructions=(
            "You are an agricultural assistant inside an app for farmers. "
            "Generate practical, safe, concise plant disease treatment guidance. "
            "Do not invent exact pesticide dosages or restricted chemical product names. "
            "If chemical treatment may be needed, say to use locally approved products according to the product label. "
            "Return plain text only, maximum 6 short sentences."
        ),
        input=(
            f"Detected plant disease label: {readable_label}. "
            "Write a treatment recommendation with immediate actions, prevention advice, monitoring advice, "
            "and safe escalation if the disease spreads. Do not answer only with 'consult an agronomist'."
        ),
        max_output_tokens=220,
    )

    recommendation = getattr(response, "output_text", "").strip()
    return recommendation or local_treatment_fallback(label)


def treatment_for(label: str) -> str:
    normalized_label = normalize_label(label)

    if normalized_label in {"healthy", "no_disease_detected", "no_disease"}:
        return local_treatment_fallback(label)

    try:
        return openai_treatment_for(normalized_label)
    except Exception:
        return local_treatment_fallback(label)


@app.post("/classify")
def classify_image(request: ClassifyRequest) -> dict[str, Any]:
    try:
        image_bytes = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}")

    try:
        results = model(image)
        detections: list[dict[str, Any]] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = str(model.names[class_id])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

                detections.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        if not detections:
            return {
                "status": "success",
                "model": MODEL_NAME,
                "disease": "No disease detected",
                "predicted_disease": "No disease detected",
                "confidence": 0.0,
                "detections": [],
                "recommendation": "Try another clearer leaf image, preferably close-up and well lit.",
            }

        best_detection = max(detections, key=lambda item: item["confidence"])
        disease = best_detection["label"]

        return {
            "status": "success",
            "model": MODEL_NAME,
            "disease": disease,
            "predicted_disease": disease,
            "confidence": best_detection["confidence"],
            "detections": detections,
            "recommendation": treatment_for(disease),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
