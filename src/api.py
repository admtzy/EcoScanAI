"""
=========================================================
EcoScan AI
FastAPI Server
=========================================================
"""

import uuid
import shutil

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from src.predictor import (
    predict,
    get_classes
)

from src.config import (
    ROOT_DIR,
    HOST,
    PORT
)

# =====================================================
# APP
# =====================================================

app = FastAPI(

    title="EcoScan AI API",

    version="1.0.0"

)

# =====================================================
# CORS
# =====================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)

# =====================================================
# UPLOAD FOLDER
# =====================================================

UPLOAD_FOLDER = ROOT_DIR / "uploads"

UPLOAD_FOLDER.mkdir(

    exist_ok=True

)

# =====================================================
# HOME
# =====================================================

@app.get("/")

def home():

    return {

        "success": True,

        "project": "EcoScan AI",

        "version": "1.0.0"

    }

# =====================================================
# HEALTH
# =====================================================

@app.get("/health")

def health():

    return {

        "success": True,

        "status": "API Ready"

    }

# =====================================================
# CLASSES
# =====================================================

@app.get("/classes")

def classes():

    return {

        "classes": get_classes()

    }

# =====================================================
# PREDICT
# =====================================================

@app.post("/predict")

async def predict_api(

    image: UploadFile = File(...),

    weight: float = Form(...)

):

    if image.content_type is None or not image.content_type.startswith("image/"):

        raise HTTPException(

            status_code=400,

            detail="File harus berupa gambar"

        )

    extension = image.filename.split(".")[-1]

    filename = f"{uuid.uuid4()}.{extension}"

    filepath = UPLOAD_FOLDER / filename

    with open(filepath, "wb") as buffer:

        shutil.copyfileobj(

            image.file,

            buffer

        )

    try:

        result = predict(

            str(filepath),

            weight

        )

        return {

            "success": True,

            "data": result

        }

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )

    finally:

        if filepath.exists():

            filepath.unlink()