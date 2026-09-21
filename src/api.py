"""
Week 3, Day 4 — FastAPI service wrapping the CV pipeline.
"""

import os
import shutil
import tempfile
import traceback
import sys

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pipeline import run_full_cv_pipeline

app = FastAPI(
    title="Auto-Adjuster CV Service",
    description="Week 1-3 deliverable: video -> segmented, classified, fused Damage JSON.",
    version="0.1.0",
)

SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "checkpoints/sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_b")
CLASSIFIER_CHECKPOINT = os.environ.get(
    "CLASSIFIER_CHECKPOINT", "checkpoints/damage_classifier_best.pth"
)


class HealthResponse(BaseModel):
    status: str
    sam_checkpoint_found: bool
    classifier_checkpoint_found: bool


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Lets an orchestrator/load-balancer check readiness before routing real traffic."""
    return HealthResponse(
        status="ok",
        sam_checkpoint_found=os.path.exists(SAM_CHECKPOINT),
        classifier_checkpoint_found=os.path.exists(CLASSIFIER_CHECKPOINT),
    )


@app.post("/cv/infer")
async def infer(
    video: UploadFile = File(..., description="Walk-around video, mp4/mov"),
    claim_id: str = Form(
        None, description="Optional claim ID; auto-generated if omitted"
    ),
):
    """Accepts a video upload, runs the full CV pipeline, returns the Damage JSON."""
    if not os.path.exists(SAM_CHECKPOINT):
        raise HTTPException(
            status_code=503,
            detail=f"SAM checkpoint not found at {SAM_CHECKPOINT}. Service not ready.",
        )
    if not os.path.exists(CLASSIFIER_CHECKPOINT):
        raise HTTPException(
            status_code=503,
            detail=f"Classifier not found at {CLASSIFIER_CHECKPOINT}. Run Week 2 first.",
        )

    suffix = os.path.splitext(video.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_video_path = tmp.name

    try:
        if "src" not in sys.path:
            sys.path.insert(0, "src")

        result = run_full_cv_pipeline(
            video_path=tmp_video_path,
            sam_checkpoint=SAM_CHECKPOINT,
            sam_model_type=SAM_MODEL_TYPE,
            classifier_checkpoint=CLASSIFIER_CHECKPOINT,
            claim_id=claim_id,
        )
        return JSONResponse(content=result)
    except Exception as e:
        err_msg = f"Pipeline failed: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=err_msg)
    finally:
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)
