"""Groq vision backend for i-Boating chart digitisation.

Mirrors :func:`backend.iboating._extract_depths_from_chart` (Gemini) but
calls Groq's OpenAI-compatible chat-completions endpoint with a multimodal
Llama-4 Scout model.  Same prompt skeleton, same output schema, so the
augmentation pipeline can swap backends without further changes.

Set ``GROQ_API_KEY`` in the environment to enable it.
"""
from __future__ import annotations

import base64 as _b64
import io
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

L = logging.getLogger("iboating_groq")
if not L.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
    L.addHandler(h)
    L.setLevel(logging.INFO)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_DEFAULT = os.environ.get(
    "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
MAX_DEPTH_M = 25.0


def _build_prompt(bbox, img_w: int, img_h: int) -> str:
    w, s, e, n = bbox
    return f"""You are a professional hydrographic analyst examining a high-resolution screenshot
of the i-Boating nautical chart application.

Image: {img_w} x {img_h} pixels.
Bounds: SW({s:.5f} N, {w:.5f} E) to NE({n:.5f} N, {e:.5f} E).

TASK: Extract ONLY genuine depth soundings from the water area of this chart.

DEPTH SOUNDINGS look like:
- Plain numbers like 3.2, 15, 7.8, 22.1 (often with one decimal)
- Printed in a small font directly on the blue water area
- Standalone, not enclosed in any symbol/circle/diamond/box
- Not underlined (underlined = drying height, ignore those)
- Units are metres (treat all numbers as metres unless feet/fathoms is obvious)

DO NOT EXTRACT:
- Buoy / marker / mooring numbers (numbers inside or attached to symbols)
- Channel / route / shipping-lane numbers
- Light characteristics (Fl 5s, Fl(2)R 10s, Iso 4s, Q(6)+LFl 15s, etc.)
- Bridge clearances or heights on land
- Distance markers, chart reference numbers, lat/lon labels
- UI text (zoom, scale bar, copyright)

DEPTH CONTOUR LABELS: numbers labelling isobath lines (2, 5, 10, 20 ...).
Mark these as type "contour".

OUTPUT — ONE JSON ARRAY, NOTHING ELSE.  Each entry:
{{"x": pixel_from_left, "y": pixel_from_top, "depth": metres, "type": "sounding"|"contour", "confidence": 0.0-1.0}}

Coordinates: x in [0, {img_w}], y in [0, {img_h}].
Depth always in METRES, positive number.
QUALITY over QUANTITY — if uncertain, exclude.
If you see no verified soundings, return [].
Return ONLY the JSON array, no commentary, no markdown fences.
"""


def _parse_json_loose(text: str) -> Optional[List[Dict]]:
    """Tolerant JSON-array parser for LLM output."""
    s = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else None
    except Exception:
        pass
    # Try slicing from first '[' to last '}' + ']'
    i = s.find("[")
    if i < 0:
        return None
    sub = s[i:]
    try:
        v = json.loads(sub)
        return v if isinstance(v, list) else None
    except Exception:
        pass
    j = sub.rfind("}")
    if j > 0:
        try:
            v = json.loads(sub[: j + 1] + "]")
            return v if isinstance(v, list) else None
        except Exception:
            return None
    return None


def extract_depths_from_chart_groq(
    img_b64: str,
    bbox: List[float],
    img_w: int,
    img_h: int,
    model: str = GROQ_MODEL_DEFAULT,
    api_key: Optional[str] = None,
    max_image_px: int = 1024,
    timeout: int = 180,
    min_confidence: float = 0.5,
) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """Send the chart image to Groq's multimodal Llama-4 and return the
    list of depth points (same schema as the Gemini variant)."""
    api_key = api_key or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None, "GROQ_API_KEY not set"

    # Groq image-payload limits are stricter than Gemini's.  Pre-resize the
    # PNG so the data-URL stays under the request size cap.
    try:
        from PIL import Image
        raw = _b64.b64decode(img_b64)
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        scale = min(1.0, max_image_px / max(im.size))
        if scale < 1.0:
            new_w = int(im.size[0] * scale)
            new_h = int(im.size[1] * scale)
            im = im.resize((new_w, new_h), Image.LANCZOS)
            buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
            img_b64 = _b64.b64encode(buf.getvalue()).decode()
            # Rescale prompt dims so the model returns coordinates in the
            # rescaled image space (we'll undo this when we map to lat/lon).
            img_w_use, img_h_use = im.size
        else:
            img_w_use, img_h_use = img_w, img_h
    except Exception as ex:
        L.warning(f"Groq i-Boating: image preprocess failed ({ex}); sending as-is")
        img_w_use, img_h_use = img_w, img_h

    prompt = _build_prompt(bbox, img_w_use, img_h_use)
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }],
        "temperature": 0.1,
        "max_tokens": 8192,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    try:
        r = requests.post(GROQ_URL, headers=headers, json=body, timeout=timeout)
    except Exception as ex:
        return None, f"Groq request failed: {ex}"
    if not r.ok:
        # response_format json_object requires the prompt to mention "json".
        # If Groq rejects with 400, retry without that constraint.
        if r.status_code == 400 and "json" in (r.text or "").lower():
            body.pop("response_format", None)
            try:
                r = requests.post(GROQ_URL, headers=headers, json=body, timeout=timeout)
            except Exception as ex:
                return None, f"Groq retry failed: {ex}"
        if not r.ok:
            return None, f"Groq {r.status_code}: {r.text[:300]}"

    try:
        text = r.json()["choices"][0]["message"]["content"]
    except Exception as ex:
        return None, f"Groq response parse failed: {ex}"

    pts = _parse_json_loose(text)
    if pts is None:
        # Some models wrap the array in {"data":[...]} or {"depths":[...]}
        try:
            obj = json.loads(text)
            for k in ("data", "depths", "soundings", "points", "items"):
                if k in obj and isinstance(obj[k], list):
                    pts = obj[k]
                    break
        except Exception:
            pass
    if not pts:
        return None, f"Groq returned no parsable depths (len={len(text)})"

    # Validate + scale back to original pixel space
    sx = img_w / max(1.0, img_w_use)
    sy = img_h / max(1.0, img_h_use)
    valid: List[Dict] = []
    for p in pts:
        try:
            d = float(p.get("depth", 0))
            x = float(p.get("x", 0)) * sx
            y = float(p.get("y", 0)) * sy
            t = str(p.get("type", "sounding"))
            conf = float(p.get("confidence", 0.7))
        except Exception:
            continue
        if not (0 < d <= MAX_DEPTH_M):
            continue
        if conf < min_confidence:
            continue
        if not (0 <= x <= img_w and 0 <= y <= img_h):
            continue
        valid.append({"x": x, "y": y, "depth": min(d, MAX_DEPTH_M),
                       "type": t, "confidence": conf})
    n_snd = sum(1 for p in valid if p["type"] == "sounding")
    n_cnt = sum(1 for p in valid if p["type"] == "contour")
    if valid:
        L.info(f"Groq: {len(valid)} verified points "
               f"({n_snd} soundings, {n_cnt} contours) — "
               f"avg confidence {np.mean([p['confidence'] for p in valid]):.2f}")
        return valid, None
    return None, f"Groq returned 0 valid soundings (raw len={len(text)})"
