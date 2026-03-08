#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import contextlib
import io
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from PIL import Image, ImageOps
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from ultralytics import YOLO
from mcp.server.fastmcp import FastMCP

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("yolo-mcp-only")

# =============================================================================
# Config
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", os.path.join(BASE_DIR, "best.pt"))
SPECIES_MAP_PATH = os.getenv("SPECIES_MAP_PATH", os.path.join(BASE_DIR, "species_map.json"))

CONF_THRES = float(os.getenv("YOLO_CONF_THRES", "0.05"))
IOU_THRES = float(os.getenv("YOLO_IOU_THRES", "0.45"))
IMG_SIZE = int(os.getenv("YOLO_IMGSZ", "640"))
DEVICE = os.getenv("YOLO_DEVICE", "cpu")
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "3"))

MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8001"))

# =============================================================================
# Utils
# =============================================================================
def _load_species_map(path: str) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(path):
        logger.warning("species_map.json not found: %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("species_map.json is not a dict. Fallback to model.names.")
            return {}
        return data
    except Exception as e:
        logger.warning("Failed to load species_map.json: %s", e)
        return {}


SPECIES_MAP = _load_species_map(SPECIES_MAP_PATH)


def _decode_base64_image(image_base64: str) -> Image.Image:
    if not image_base64 or not isinstance(image_base64, str):
        raise ValueError("image_base64 must be a non-empty string.")

    # data:image/png;base64,... 형태도 허용
    if image_base64.startswith("data:") and "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_base64)
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}") from e

    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Failed to decode image from base64: {e}") from e


def _load_image_from_url(image_url: str) -> Image.Image:
    if not image_url or not isinstance(image_url, str):
        raise ValueError("image_url must be a non-empty string.")

    req = urllib.request.Request(
        image_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Failed to download image from URL (HTTP {e.code}).") from e
    except Exception as e:
        raise RuntimeError(f"Failed to download image from URL: {e}") from e

    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img
    except Exception as e:
        raise RuntimeError(f"Failed to parse image downloaded from URL: {e}") from e


def _image_to_numpy_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img)


def _xyxy_to_list(xyxy: np.ndarray) -> List[float]:
    return [float(x) for x in xyxy.tolist()]


def _get_species_info(class_id: int, model_names: Dict[int, str]) -> Dict[str, Optional[str]]:
    key = str(class_id)
    info = SPECIES_MAP.get(key, {})

    scientific_name = info.get("scientific_name") or model_names.get(class_id)
    korean_name = info.get("korean_name")
    common_name = info.get("common_name")

    return {
        "scientific_name": scientific_name,
        "korean_name": korean_name,
        "common_name": common_name,
    }


# =============================================================================
# Model Load
# =============================================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}. "
        f"Put best.pt in the server directory or set YOLO_MODEL_PATH."
    )

logger.info("Loading YOLO model from: %s", MODEL_PATH)
model = YOLO(MODEL_PATH)
logger.info("YOLO model loaded successfully")

# =============================================================================
# Core Detection
# =============================================================================
def _run_detection(pil_img: Image.Image, top_k: int, input_type: str) -> Dict[str, Any]:
    width, height = pil_img.size
    img_np = _image_to_numpy_rgb(pil_img)

    results = model.predict(
        source=img_np,
        conf=CONF_THRES,
        iou=IOU_THRES,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False,
    )

    if not results or len(results) == 0:
        return {
            "ok": True,
            "input_type": input_type,
            "image_size": {"width": width, "height": height},
            "num_detections": 0,
            "predicted_label": None,
            "best": None,
            "detections": [],
            "notes": "YOLO returned no results."
        }

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return {
            "ok": True,
            "input_type": input_type,
            "image_size": {"width": width, "height": height},
            "num_detections": 0,
            "predicted_label": None,
            "best": None,
            "detections": [],
            "notes": f"No detections above threshold (conf={CONF_THRES})."
        }

    confs = boxes.conf.detach().cpu().numpy()
    clss = boxes.cls.detach().cpu().numpy().astype(int)
    xyxys = boxes.xyxy.detach().cpu().numpy()

    order = np.argsort(-confs)
    order = order[:max(1, int(top_k))]

    names_map = result.names if hasattr(result, "names") else getattr(model, "names", {})

    detections: List[Dict[str, Any]] = []
    for rank, idx in enumerate(order, start=1):
        class_id = int(clss[idx])
        confidence = float(confs[idx])
        bbox_xyxy = _xyxy_to_list(xyxys[idx])
        species_info = _get_species_info(class_id, names_map)

        detections.append({
            "rank": rank,
            "class_id": class_id,
            "confidence": confidence,
            "bbox_xyxy": bbox_xyxy,
            **species_info
        })

    best = detections[0] if detections else None

    predicted_label = None
    if best is not None:
        predicted_label = (
            best.get("korean_name")
            or best.get("scientific_name")
            or best.get("common_name")
            or f"class_{best['class_id']}"
        )

    logger.info(
        "Prediction finished | input_type=%s | num_detections=%s | predicted_label=%s",
        input_type,
        len(confs),
        predicted_label,
    )

    return {
        "ok": True,
        "input_type": input_type,
        "image_size": {"width": width, "height": height},
        "num_detections": int(len(confs)),
        "predicted_label": predicted_label,
        "best": best,
        "detections": detections,
        "notes": "Returned strictly based on YOLO detection results."
    }


# =============================================================================
# MCP Server
# =============================================================================
mcp = FastMCP("yolo11-species-mcp", json_response=True)

# 구버전 호환
if hasattr(mcp, "settings") and hasattr(mcp.settings, "streamable_http_path"):
    mcp.settings.streamable_http_path = "/"


@mcp.tool()
def ping() -> Dict[str, Any]:
    """Simple connectivity test."""
    logger.info("ping called")
    return {"ok": True, "message": "pong"}


@mcp.tool()
def detect_species_from_base64(
    image_base64: str,
    top_k: int = TOP_K_DEFAULT,
) -> Dict[str, Any]:
    """
    Detect species from a base64 encoded image.

    Preferred MCP tool for image classification.
    Use a raw base64 string or a data URL.
    """
    logger.info(
        "detect_species_from_base64 called | has_base64=%s | top_k=%s",
        bool(image_base64),
        top_k,
    )
    try:
        pil_img = _decode_base64_image(image_base64)
        return _run_detection(
            pil_img=pil_img,
            top_k=max(1, int(top_k)),
            input_type="image_base64",
        )
    except Exception as e:
        logger.exception("detect_species_from_base64 failed")
        return {
            "ok": False,
            "predicted_label": None,
            "best": None,
            "detections": [],
            "notes": f"{type(e).__name__}: {e}"
        }


@mcp.tool()
def detect_species_from_url(
    image_url: str,
    top_k: int = TOP_K_DEFAULT,
) -> Dict[str, Any]:
    """
    Detect species from a public image URL.

    Use only if the image is publicly downloadable.
    Private blob URLs or signed URLs may fail.
    """
    logger.info(
        "detect_species_from_url called | image_url=%s | top_k=%s",
        image_url,
        top_k,
    )
    try:
        pil_img = _load_image_from_url(image_url)
        return _run_detection(
            pil_img=pil_img,
            top_k=max(1, int(top_k)),
            input_type="image_url",
        )
    except Exception as e:
        logger.exception("detect_species_from_url failed")
        return {
            "ok": False,
            "predicted_label": None,
            "best": None,
            "detections": [],
            "notes": f"{type(e).__name__}: {e}"
        }


@mcp.tool()
def detect_species_from_image(
    image_base64: Optional[str] = None,
    image_url: Optional[str] = None,
    top_k: int = TOP_K_DEFAULT,
) -> Dict[str, Any]:
    """
    Generic image detection tool.

    Provide exactly one of:
    - image_base64
    - image_url

    Prefer image_base64 whenever possible.
    """
    try:
        logger.info(
            "detect_species_from_image called | has_base64=%s | image_url=%s | top_k=%s",
            bool(image_base64),
            image_url,
            top_k,
        )

        provided = [
            image_base64 is not None and str(image_base64).strip() != "",
            image_url is not None and str(image_url).strip() != "",
        ]

        if sum(provided) != 1:
            return {
                "ok": False,
                "predicted_label": None,
                "best": None,
                "detections": [],
                "notes": "Provide exactly one of image_base64 or image_url."
            }

        top_k = max(1, int(top_k))

        if image_base64:
            pil_img = _decode_base64_image(image_base64)
            return _run_detection(pil_img, top_k, "image_base64")

        pil_img = _load_image_from_url(image_url)  # type: ignore[arg-type]
        return _run_detection(pil_img, top_k, "image_url")

    except Exception as e:
        logger.exception("detect_species_from_image failed")
        return {
            "ok": False,
            "predicted_label": None,
            "best": None,
            "detections": [],
            "notes": f"{type(e).__name__}: {e}"
        }


# =============================================================================
# Optional HTTP routes for health/debug
# =============================================================================
async def root_handler(request):
    return JSONResponse({
        "ok": True,
        "message": "MCP server is running",
        "routes": ["/health", "/mcp"],
        "tools": [
            "ping",
            "detect_species_from_base64",
            "detect_species_from_url",
            "detect_species_from_image",
        ],
    })


async def health_handler(request):
    return JSONResponse({
        "ok": True,
        "message": "server alive",
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "mcp_port": MCP_PORT,
    })


# =============================================================================
# Run
# =============================================================================
def run_mcp() -> None:
    logger.info("Starting MCP server on %s:%s", MCP_HOST, MCP_PORT)

    @contextlib.asynccontextmanager
    async def lifespan(app_: Starlette):
        async with mcp.session_manager.run():
            yield

    mcp_app = Starlette(
        routes=[
            Route("/", endpoint=root_handler, methods=["GET"]),
            Route("/health", endpoint=health_handler, methods=["GET"]),
            Mount("/mcp", app=mcp.streamable_http_app()),
        ],
        lifespan=lifespan,
    )

    uvicorn.run(
        mcp_app,
        host=MCP_HOST,
        port=MCP_PORT,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    run_mcp()