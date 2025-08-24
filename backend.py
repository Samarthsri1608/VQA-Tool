import os
import io
import base64
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

from ultralytics import YOLO  # pip install ultralytics
from google import genai      # pip install google-genai
from google.genai import types
import wave
# ------------------------
# Config
# ------------------------
# Expect GOOGLE_API_KEY in env
GOOGLE_API_KEY = "AIzaSyCTTCnoBPJBrKLxOmxcMQZIznVor3r3QdM"
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not set. Set it before running or Gemini calls will fail.")

app = Flask(__name__)
CORS(app)

# YOLOv8s (downloaded automatically on first run)
yolo = YOLO("yolov8s.pt")

# Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)

ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}


# ------------------------
# Helpers
# ------------------------
def allowed_file(fname: str) -> bool:
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def img_to_b64(pil_img: Image.Image) -> str:
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG", quality=90)
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """Pillow 10+ removed textsize; use textbbox for compatibility."""
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    except Exception:
        # Fallback for very old Pillow
        return draw.textsize(text, font=font)

def draw_boxes(pil_img: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['label']} {det['confidence']:.2f}"

        # box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # label background
        tw, th = text_size(draw, label, font)
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 8, y1], fill="red")
        # text
        draw.text((x1 + 4, y1 - th - 4), label, fill="white", font=font)

    return img


# ------------------------
# Routes
# ------------------------
@app.route("/")
def home():
    # Serve the frontend if placed alongside app.py
    return send_file("web.html")


@app.route("/detect", methods=["POST"])
def detect():
    """
    Form-data:
      - image: file
      - question (optional): string (VQA question)
    Returns JSON:
      - detections: [{label, confidence, box:[x1,y1,x2,y2]}]
      - annotated_image_b64: base64 JPEG (no prefix)
      - answer: VQA answer string
    """
    if "image" not in request.files:
        return jsonify({"error": "no image file provided"}), 400

    f = request.files["image"]
    if f.filename == "" or not allowed_file(f.filename):
        return jsonify({"error": "invalid or unsupported file"}), 400

    question = request.form.get("question", "").strip()

    raw = f.read()
    pil = pil_from_bytes(raw)
    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # --- YOLOv8s inference ---
    res = yolo.predict(img_bgr, imgsz=640, conf=0.25, verbose=False)[0]

    detections = []
    for b in res.boxes:
        xyxy = b.xyxy.cpu().numpy()[0].tolist()  # [x1,y1,x2,y2]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        conf = float(b.conf.cpu().numpy()[0])
        cls = int(b.cls.cpu().numpy()[0])
        label = yolo.names[cls] if hasattr(yolo, "names") else str(cls)

        detections.append({
            "label": label,
            "confidence": conf,
            "box": [x1, y1, x2, y2]
        })

    annotated = draw_boxes(pil, detections)
    annotated_b64 = img_to_b64(annotated)

    # --- Build VQA prompt ---
    # We pass both the detections summary and the user's question to Gemini, alongside the image itself.
    summary_lines = [f"- {d['label']} (conf {d['confidence']:.2f}) at {d['box']}" for d in detections] or ["- No objects detected"]
    det_summary = "Objects detected by YOLO:\n" + "\n".join(summary_lines)

    user_q = question if question else "Provide a concise description and count of key objects."
    prompt_text = (
        "You are a visual question answering assistant.\n"
        f"{det_summary}\n\n"
        f"User question: {user_q}\n"
        "Answer clearly and concisely, referencing visible objects if relevant."
    )

    # --- Gemini multimodal call (image + text) ---
    answer_text = ""
    try:
        # The Python client accepts PIL.Image as a content item in many versions.
        # If your version requires explicit parts, adapt accordingly.
        g_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pil, prompt_text]
        )
        # .text is available on successful responses
        answer_text = getattr(g_response, "text", "") or ""
    except Exception as e:
        # If Gemini fails, still return detections and annotated image
        answer_text = f"(VQA error) {e}"

    return jsonify({
        "detections": detections,
        "annotated_image_b64": annotated_b64,
        "answer": answer_text
    })


# @app.route("/tts", methods=["POST"])
# def tts():
#     """
#     JSON:
#       - text: string
#       - (optional) voice_name: string (default: 'Kore')
#     Returns: WAV audio bytes stream
#     """
#     try:
#         data = request.get_json(silent=True) or {}
#         text = (data.get("text") or "").strip()
#         voice_name = data.get("voice_name") or "Kore"

#         if not text:
#             return jsonify({"error": "No text provided"}), 400

#         # Gemini TTS
#         tts_resp = client.models.generate_content(
#             model="gemini-2.5-flash-preview-tts",
#             contents=text,
#             config=types.GenerateContentConfig(
#                 response_modalities=["AUDIO"],
#                 speech_config=types.SpeechConfig(
#                     voice_config=types.VoiceConfig(
#                         prebuilt_voice_config=types.PrebuiltVoiceConfig(
#                             voice_name=voice_name
#                         )
#                     )
#                 ),
#             )
#         )

#         # Decode base64 PCM bytes
#         data_b64 = tts_resp.candidates[0].content.parts[0].inline_data.data
#         pcm_bytes = base64.b64decode(data_b64)

#         # Wrap PCM16 in WAV in memory
#         buffer = io.BytesIO()
#         with wave.open(buffer, "wb") as wf:
#             wf.setnchannels(1)        # mono
#             wf.setsampwidth(2)        # 16-bit PCM
#             wf.setframerate(24000)    # 24kHz as Gemini default
#             wf.writeframes(pcm_bytes)
#         buffer.seek(0)

#         return Response(buffer.read(), mimetype="audio/wav")

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
