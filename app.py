"""
Face Recognition System using Machine Learning
================================================
A beginner-friendly Flask web application that trains machine-learning models
on the Olivetti Faces dataset and exposes REST endpoints for prediction.

Models compared:
    - Gaussian Naive Bayes
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)

The best-performing model (selected via cross-validation) is saved to
``model/saved_model.pkl`` and used for predictions on uploaded images.
"""

import io
import os
import pickle
import logging

import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import cv2  # opencv-python-headless

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path where the trained model pipeline will be saved / loaded from.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "saved_model.pkl")

# Number of PCA components to keep during dimensionality reduction.
N_PCA_COMPONENTS = 150

# Number of cross-validation folds used when evaluating models.
CV_FOLDS = 5

# Olivetti face image dimensions (64×64 pixels).
IMG_SIZE = 64

# Allowed image extensions for uploads.
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ---------------------------------------------------------------------------
# Global state – populated by ``load_or_train_model()``
# ---------------------------------------------------------------------------

best_pipeline = None          # sklearn Pipeline (scaler → PCA → classifier)
model_accuracies = {}         # {"Gaussian Naive Bayes": 0.93, …}
best_model_name = ""          # e.g. "SVM"
dataset_labels = None         # Olivetti target labels (0–39)
dataset_images = None         # Olivetti images (400, 4096)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def allowed_file(filename: str) -> bool:
    """Return *True* if ``filename`` has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes into the 1-D feature vector the model expects.

    Steps
    -----
    1. Open with PIL and convert to grayscale (``"L"``).
    2. Resize to 64×64 pixels (Olivetti face size).
    3. Convert to a NumPy float array and normalize to [0, 1].
    4. Flatten to a 1-D vector of length 4096.

    Returns
    -------
    np.ndarray
        Shape ``(1, 4096)`` – ready to pass into ``best_pipeline.predict()``.
    """
    # Open image from raw bytes
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Resize using OpenCV for high-quality interpolation
    img_array = np.array(pil_image, dtype=np.float64)
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Normalize pixel values to [0, 1] (same as Olivetti dataset)
    img_normalized = img_resized / 255.0

    # Flatten to a 1-D vector and wrap in a 2-D array for sklearn
    return img_normalized.flatten().reshape(1, -1)


# ---------------------------------------------------------------------------
# Model training & persistence
# ---------------------------------------------------------------------------


def train_models():
    """Train multiple ML models on the Olivetti Faces dataset.

    Workflow
    --------
    1. Fetch the Olivetti Faces dataset (400 images of 40 people).
    2. Normalize features and reduce dimensions with PCA.
    3. Train Gaussian NB, SVM, and KNN using *k*-fold cross-validation.
    4. Select the model with the highest mean CV accuracy.
    5. Fit the winning pipeline on the full training set.
    6. Save the pipeline to ``MODEL_PATH`` with :mod:`pickle`.

    Returns
    -------
    tuple
        ``(best_pipeline, model_accuracies, best_model_name, X, y)``
    """
    logger.info("Fetching Olivetti Faces dataset …")
    data = fetch_olivetti_faces()
    X, y = data.data, data.target  # X shape: (400, 4096), y shape: (400,)
    logger.info("Dataset loaded: %d samples, %d features.", X.shape[0], X.shape[1])

    # ------------------------------------------------------------------
    # Define candidate models.  Each pipeline normalises the features,
    # applies PCA, then feeds into the classifier.
    # ------------------------------------------------------------------
    candidates = {
        "Gaussian Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    accuracies = {}
    best_score = -1.0
    best_name = ""
    best_pipe = None

    for name, clf in candidates.items():
        # Build a pipeline: StandardScaler → PCA → Classifier
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=N_PCA_COMPONENTS, random_state=42)),
            ("classifier", clf),
        ])

        # Evaluate with cross-validation
        scores = cross_val_score(pipe, X, y, cv=CV_FOLDS, scoring="accuracy")
        mean_acc = float(np.mean(scores))
        accuracies[name] = round(mean_acc, 4)

        logger.info(
            "Model %-25s | CV accuracy: %.4f (±%.4f)",
            name, mean_acc, float(np.std(scores)),
        )

        # Track the best model
        if mean_acc > best_score:
            best_score = mean_acc
            best_name = name
            best_pipe = pipe

    # ------------------------------------------------------------------
    # Retrain the best pipeline on the *entire* dataset so the saved
    # model has seen all available data.
    # ------------------------------------------------------------------
    logger.info("Best model: %s (%.4f). Training on full dataset …", best_name, best_score)
    best_pipe.fit(X, y)

    # ------------------------------------------------------------------
    # Save to disk
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "pipeline": best_pipe,
            "accuracies": accuracies,
            "best_model": best_name,
        }, f)
    logger.info("Model saved to %s", MODEL_PATH)

    return best_pipe, accuracies, best_name, X, y


def load_or_train_model():
    """Load a previously saved model or train from scratch if absent.

    This is called once at application startup.  It populates the module-level
    globals ``best_pipeline``, ``model_accuracies``, ``best_model_name``,
    ``dataset_images``, and ``dataset_labels``.
    """
    global best_pipeline, model_accuracies, best_model_name
    global dataset_images, dataset_labels

    if os.path.exists(MODEL_PATH):
        logger.info("Loading saved model from %s …", MODEL_PATH)
        with open(MODEL_PATH, "rb") as f:
            saved = pickle.load(f)
        best_pipeline = saved["pipeline"]
        model_accuracies = saved["accuracies"]
        best_model_name = saved["best_model"]

        # We still need the dataset reference for label count / metadata
        data = fetch_olivetti_faces()
        dataset_images, dataset_labels = data.data, data.target
        logger.info("Model loaded successfully (%s).", best_model_name)
    else:
        logger.info("No saved model found – training from scratch …")
        best_pipeline, model_accuracies, best_model_name, dataset_images, dataset_labels = (
            train_models()
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Render the main UI page."""
    return render_template("index.html")


@app.route("/api/models", methods=["GET"])
def api_models():
    """Return model accuracy comparison as JSON.

    Response example::

        {
            "models": {
                "Gaussian Naive Bayes": 0.8275,
                "SVM": 0.9550,
                "KNN": 0.9350
            },
            "best_model": "SVM"
        }
    """
    return jsonify({
        "models": model_accuracies,
        "best_model": best_model_name,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Accept an uploaded image and return the predicted person label + confidence.

    Expects
    -------
    multipart/form-data with a field named ``image``.

    Returns
    -------
    JSON with ``predicted_label``, ``confidence``, and ``model_used``.
    On error returns a JSON body with an ``error`` key and an appropriate
    HTTP status code.
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

    # ---- preprocess -----------------------------------------------------
    try:
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({"error": "Uploaded file is empty."}), 400

        processed = preprocess_image(image_bytes)
    except Exception:
        logger.exception("Image preprocessing failed.")
        return jsonify({"error": "Could not process image. Please upload a valid image file."}), 400

    # ---- predict --------------------------------------------------------
    try:
        predicted_label = int(best_pipeline.predict(processed)[0])

        # Confidence score
        classifier = best_pipeline.named_steps["classifier"]
        if hasattr(classifier, "predict_proba"):
            # Use predict_proba when available (SVM with probability=True, NB, KNN)
            proba = best_pipeline.predict_proba(processed)[0]
            confidence = float(np.max(proba))
        elif hasattr(classifier, "decision_function"):
            # Fallback: normalise the decision-function output to [0, 1]
            decision = best_pipeline.decision_function(processed)[0]
            # For multi-class SVM decision_function returns an array
            if isinstance(decision, np.ndarray):
                confidence = float(
                    np.exp(np.max(decision)) / np.sum(np.exp(decision))
                )
            else:
                confidence = float(1.0 / (1.0 + np.exp(-decision)))
        else:
            # Last resort – report a fixed placeholder
            confidence = None

        return jsonify({
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "model_used": best_model_name,
            "message": f"Predicted as Person #{predicted_label}",
        })
    except Exception:
        logger.exception("Prediction failed.")
        return jsonify({"error": "Prediction failed. Please try again with a different image."}), 500


# ---------------------------------------------------------------------------
# Application entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Train / load model before the first request is served.
    load_or_train_model()

    logger.info("Starting Flask development server on http://localhost:5000")
    # Set FLASK_DEBUG=1 in your environment to enable the interactive debugger.
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
