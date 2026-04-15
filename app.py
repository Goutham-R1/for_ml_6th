"""
Image Recognition System
=========================
A beginner-friendly Flask web application that uses OpenCV to detect and
identify content in uploaded images.

Detection capabilities:
    - Human Face detection  (OpenCV Haar Cascade)
    - Cat Face detection    (OpenCV Haar Cascade)
    - Human Body detection  (OpenCV Haar Cascade)
    - Basic Shape detection (Circle, Rectangle, Triangle, etc.)
    - Blank / empty image detection

When a human face is found the app says *"It's a Human Face"*.
When a common object or shape is found it reports the name.
When nothing is recognisable it says *"Not found anything"*.
"""

import io
import os
import logging

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import cv2  # opencv-python-headless

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

# Minimum standard-deviation of pixel values to consider an image non-blank.
BLANK_STD_THRESHOLD = 10

# Minimum fraction of edge pixels to consider the image as containing content.
EDGE_DENSITY_THRESHOLD = 0.05

# Shape-detection tuning constants
MIN_CONTOUR_AREA_RATIO = 0.01   # ignore contours smaller than 1 % of image
MAX_CONTOUR_AREA_RATIO = 0.95   # ignore contours larger than 95 % of image
APPROX_POLY_EPSILON = 0.04      # polygon approximation tolerance (fraction of arc)
SQUARE_ASPECT_MIN = 0.85        # aspect ratio range that counts as a square
SQUARE_ASPECT_MAX = 1.15
CIRCULARITY_THRESHOLD = 0.7     # minimum circularity to call a contour a circle

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ---------------------------------------------------------------------------
# OpenCV Haar-Cascade classifiers (ship with opencv-python-headless)
# ---------------------------------------------------------------------------

_cascade_dir = cv2.data.haarcascades

face_cascade = cv2.CascadeClassifier(
    os.path.join(_cascade_dir, "haarcascade_frontalface_default.xml")
)
face_cascade_alt = cv2.CascadeClassifier(
    os.path.join(_cascade_dir, "haarcascade_frontalface_alt2.xml")
)
eye_cascade = cv2.CascadeClassifier(
    os.path.join(_cascade_dir, "haarcascade_eye.xml")
)
cat_cascade = cv2.CascadeClassifier(
    os.path.join(_cascade_dir, "haarcascade_frontalcatface_extended.xml")
)
body_cascade = cv2.CascadeClassifier(
    os.path.join(_cascade_dir, "haarcascade_fullbody.xml")
)
upper_body_cascade = cv2.CascadeClassifier(
    os.path.join(_cascade_dir, "haarcascade_upperbody.xml")
)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def allowed_file(filename: str) -> bool:
    """Return *True* if *filename* has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _detect_faces(gray: np.ndarray):
    """Try two Haar cascades and return face rectangles."""
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        faces = face_cascade_alt.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
    return faces


def _detect_eyes(gray: np.ndarray):
    return eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)


def _detect_cats(gray: np.ndarray):
    return cat_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
    )


def _detect_bodies(gray: np.ndarray):
    bodies = body_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50)
    )
    if len(bodies) == 0:
        bodies = upper_body_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50)
        )
    return bodies


def _detect_shapes(gray: np.ndarray):
    """Return a list of shape names found via contour analysis."""
    shapes_found: list[str] = []

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = gray.shape[0] * gray.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < img_area * MIN_CONTOUR_AREA_RATIO or area > img_area * MAX_CONTOUR_AREA_RATIO:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        approx = cv2.approxPolyDP(contour, APPROX_POLY_EPSILON * perimeter, True)
        n_verts = len(approx)

        if n_verts == 3:
            shapes_found.append("Triangle")
        elif n_verts == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = float(w) / h if h != 0 else 0
            shapes_found.append("Square" if SQUARE_ASPECT_MIN <= aspect <= SQUARE_ASPECT_MAX else "Rectangle")
        elif n_verts == 5:
            shapes_found.append("Pentagon")
        elif n_verts > 6:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            shapes_found.append("Circle" if circularity > CIRCULARITY_THRESHOLD else "Object")

    return shapes_found


def _is_blank(gray: np.ndarray) -> bool:
    """An image is *blank* when pixel intensities barely vary."""
    return float(np.std(gray)) < BLANK_STD_THRESHOLD


# ---------------------------------------------------------------------------
# Main analysis entry-point
# ---------------------------------------------------------------------------


def analyze_image(image_bytes: bytes) -> dict:
    """Analyse raw image bytes and return a detection result dict."""
    pil_image = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(pil_image)

    # Convert to greyscale for all detectors
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # ---- Blank check ---------------------------------------------------
    if _is_blank(gray):
        return {
            "detected": False,
            "category": "nothing",
            "label": "Not Found",
            "message": "Not found anything — the image appears to be blank or empty",
            "confidence": 0.0,
            "details": "No recognisable content detected in the image",
        }

    results: list[dict] = []

    # 1. Human faces
    faces = _detect_faces(gray)
    if len(faces) > 0:
        eyes = _detect_eyes(gray)
        confidence = min(0.95, 0.70 + len(faces) * 0.10 + len(eyes) * 0.05)
        count_text = f"{len(faces)} face(s)" if len(faces) > 1 else "a face"
        eye_note = f" and {len(eyes)} eye(s)" if len(eyes) > 0 else ""
        results.append({
            "detected": True,
            "category": "human_face",
            "label": "Human Face",
            "message": f"It's a Human Face! Detected {count_text} in the image",
            "confidence": confidence,
            "details": f"Found {len(faces)} human face(s){eye_note}",
        })

    # 2. Cat faces
    cats = _detect_cats(gray)
    if len(cats) > 0:
        results.append({
            "detected": True,
            "category": "cat",
            "label": "Cat",
            "message": f"It's a Cat! Detected {len(cats)} cat face(s) in the image",
            "confidence": 0.75,
            "details": f"Found {len(cats)} cat face(s)",
        })

    # 3. Geometric shapes
    shapes = _detect_shapes(gray)
    if shapes and not results:
        unique = sorted(set(shapes))
        shape_names = ", ".join(unique)
        results.append({
            "detected": True,
            "category": "shape",
            "label": unique[0] if len(unique) == 1 else "Shapes",
            "message": f"Detected shape(s): {shape_names}",
            "confidence": 0.60,
            "details": f"Found shapes: {shape_names}",
        })

    # 4. Human body (only when no face *and* no shape was found – the
    #    body cascade is prone to false positives on simple shapes)
    if not results and len(faces) == 0:
        bodies = _detect_bodies(gray)
        if len(bodies) > 0:
            results.append({
                "detected": True,
                "category": "human_body",
                "label": "Human Body",
                "message": "Detected a human body / person in the image",
                "confidence": 0.65,
                "details": f"Found {len(bodies)} human body/bodies",
            })

    # 5. Generic content check via edge density
    if not results:
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / edges.size

        if edge_density > EDGE_DENSITY_THRESHOLD:
            results.append({
                "detected": True,
                "category": "unknown_object",
                "label": "Unknown Object",
                "message": "Object Detected — could not identify the specific object",
                "confidence": 0.30,
                "details": "The image contains content but it could not be specifically identified",
            })
        else:
            results.append({
                "detected": False,
                "category": "nothing",
                "label": "Not Found",
                "message": "Not found anything — no recognisable objects in the image",
                "confidence": 0.0,
                "details": "The image does not contain any clearly recognisable objects",
            })

    # Return the highest-confidence hit
    return max(results, key=lambda r: r["confidence"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Render the main UI page."""
    return render_template("index.html")


@app.route("/api/info", methods=["GET"])
def api_info():
    """Return the list of detection capabilities."""
    return jsonify({
        "capabilities": [
            {
                "name": "Human Face Detection",
                "description": "Detects human faces using Haar Cascade classifier",
                "icon": "bi-person-circle",
            },
            {
                "name": "Cat Detection",
                "description": "Detects cat faces in images",
                "icon": "bi-heart-fill",
            },
            {
                "name": "Human Body Detection",
                "description": "Detects full or upper human body",
                "icon": "bi-person-standing",
            },
            {
                "name": "Shape Detection",
                "description": "Identifies circles, rectangles, triangles, and more",
                "icon": "bi-pentagon",
            },
            {
                "name": "Blank Detection",
                "description": "Identifies blank or empty images",
                "icon": "bi-x-circle",
            },
        ]
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Accept an uploaded image and return what was detected.

    Expects
    -------
    multipart/form-data with a field named ``image``.

    Returns
    -------
    JSON with ``label``, ``message``, ``confidence``, ``category``, and
    ``details``.  On error a JSON body with an ``error`` key and an
    appropriate HTTP status code.
    """
    # ---- validate upload ------------------------------------------------
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Please upload an image."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected. Please choose an image."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": (
                f"Invalid file type. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
        }), 400

    # ---- analyse --------------------------------------------------------
    try:
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({"error": "Uploaded file is empty."}), 400

        result = analyze_image(image_bytes)

        return jsonify({
            "detected": result["detected"],
            "category": result["category"],
            "label": result["label"],
            "message": result["message"],
            "confidence": round(result["confidence"], 4),
            "details": result["details"],
        })
    except Exception:
        logger.exception("Image analysis failed.")
        return jsonify({
            "error": "Could not analyse the image. Please try again with a different image."
        }), 500


# ---------------------------------------------------------------------------
# Application entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting Image Recognition System on http://localhost:5000")
    # Set FLASK_DEBUG=1 in your environment to enable the interactive debugger.
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
