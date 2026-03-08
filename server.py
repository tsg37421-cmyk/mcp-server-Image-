import base64
import io
import json
import os
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
from PIL import Image

from ultralytics import YOLO

from mcp.server.fastmcp import FastMCP


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "best.pt")
SPECIES_MAP_PATH = os.getenv("SPECIES_MAP_PATH", "species_map.json")

# confidence threshold (필요시 환경변수로 조정)
CONF_THRES = float(os.getenv("YOLO_CONF_THRES", "0.25"))
IOU_THRES = float(os.getenv("YOLO_IOU_THRES", "0.45"))


def _load_species_map(path: str) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


SPECIES_MAP = _load_species_map(SPECIES_MAP_PATH)


def _pil_from_base64(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _pil_from_path(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _pil_from_url(url: str) -> Image.Image:
    # requests를 requirements에 추가하지 않아도 되게, urllib 사용
    import urllib.request

    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def _image_to_numpy_rgb(img: Image.Image) -> np.ndarray:
    # ultralytics는 PIL도 받지만, 일관성 위해 numpy RGB로 통일
    return np.array(img)


def _xyxy_to_list(xyxy: np.ndarray) -> List[float]:
    return [float(x) for x in xyxy.tolist()]


def _class_name_from_map(class_id: int) -> Dict[str, Optional[str]]:
    key = str(class_id)
    info = SPECIES_MAP.get(key, {})
    return {
        "scientific_name": info.get("scientific_name"),
        "korean_name": info.get("korean_name"),
        "common_name": info.get("common_name"),
    }


# -----------------------------------------------------------------------------
# Load model once (server startup)
# -----------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}. "
        f"Put best.pt at repo root or set YOLO_MODEL_PATH env."
    )

model = YOLO(MODEL_PATH)

# -----------------------------------------------------------------------------
# MCP Server
# -----------------------------------------------------------------------------
mcp = FastMCP("yolo11-species-mcp")


@mcp.tool()
def classify_species(
    image_base64: Optional[str] = None,
    image_url: Optional[str] = None,
    image_path: Optional[str] = None,
    top_k: int = 1,
) -> Dict[str, Any]:
    """
    YOLO11(best.pt) 기반 종 판별.
    - ChatGPT 추론/보정 없이 YOLO 결과(confidence) 기반으로 반환.
    - image_base64 / image_url / image_path 중 하나를 제공해야 함.

    Returns:
      {
        "ok": bool,
        "model_path": str,
        "top_k": int,
        "detections": [
          {
            "rank": int,
            "class_id": int,
            "confidence": float,
            "bbox_xyxy": [x1,y1,x2,y2],
            "scientific_name": str | null,
            "korean_name": str | null,
            "common_name": str | null
          }, ...
        ],
        "best": {...} | null,
        "notes": str
      }
    """
    # --------------------------
    # Validate input
    # --------------------------
    provided = [image_base64 is not None, image_url is not None, image_path is not None]
    if sum(provided) != 1:
        return {
            "ok": False,
            "model_path": MODEL_PATH,
            "top_k": top_k,
            "detections": [],
            "best": None,
            "notes": "Provide exactly one of image_base64, image_url, image_path."
        }

    # --------------------------
    # Load image
    # --------------------------
    try:
        if image_base64 is not None:
            pil = _pil_from_base64(image_base64)
        elif image_url is not None:
            pil = _pil_from_url(image_url)
        else:
            pil = _pil_from_path(image_path)  # type: ignore[arg-type]
    except Exception as e:
        return {
            "ok": False,
            "model_path": MODEL_PATH,
            "top_k": top_k,
            "detections": [],
            "best": None,
            "notes": f"Failed to load image: {e}"
        }

    img_np = _image_to_numpy_rgb(pil)

    # --------------------------
    # YOLO inference
    # --------------------------
    # ultralytics YOLO: results = model.predict(source, conf=..., iou=..., verbose=False)
    try:
        results = model.predict(
            img_np,
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=False
        )
    except Exception as e:
        return {
            "ok": False,
            "model_path": MODEL_PATH,
            "top_k": top_k,
            "detections": [],
            "best": None,
            "notes": f"YOLO inference failed: {e}"
        }

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return {
            "ok": True,
            "model_path": MODEL_PATH,
            "top_k": top_k,
            "detections": [],
            "best": None,
            "notes": "No detections above threshold."
        }

    boxes = results[0].boxes
    # boxes.conf, boxes.cls, boxes.xyxy
    confs = boxes.conf.detach().cpu().numpy()
    clss = boxes.cls.detach().cpu().numpy().astype(int)
    xyxys = boxes.xyxy.detach().cpu().numpy()

    # --------------------------
    # Sort by confidence desc
    # --------------------------
    order = np.argsort(-confs)
    top_k = max(1, int(top_k))
    order = order[:top_k]

    detections: List[Dict[str, Any]] = []
    for rank, idx in enumerate(order, start=1):
        class_id = int(clss[idx])
        confidence = float(confs[idx])
        bbox_xyxy = _xyxy_to_list(xyxys[idx])

        name_info = _class_name_from_map(class_id)

        detections.append({
            "rank": rank,
            "class_id": class_id,
            "confidence": confidence,
            "bbox_xyxy": bbox_xyxy,
            **name_info
        })

    best = detections[0] if detections else None

    # NOTE: 여기서 “LLM이 보정해서 바꾸지 않도록”
    #       결과는 YOLO 출력 기반으로만 작성 (추가 추론/재해석 X)
    return {
        "ok": True,
        "model_path": MODEL_PATH,
        "top_k": top_k,
        "detections": detections,
        "best": best,
        "notes": "Returned strictly based on YOLO11 predictions (no LLM override)."
    }


if __name__ == "__main__":
    # FastMCP 서버 실행
    # 실행: python server.py
    mcp.run()
