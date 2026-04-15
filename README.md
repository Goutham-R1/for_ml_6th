# Face Recognition System using Machine Learning

A beginner-friendly web application that uses **scikit-learn** and **Flask** to
recognise faces from the classic
[Olivetti Faces](https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces)
dataset.  Three models are trained and compared — the best one is automatically
selected and served through a clean web UI.

---

## Features

| Category | Details |
|----------|---------|
| **ML Models** | Gaussian Naive Bayes, SVM (RBF kernel), K-Nearest Neighbors |
| **Preprocessing** | StandardScaler normalisation → PCA (150 components) |
| **Evaluation** | 5-fold cross-validation with accuracy comparison |
| **Prediction** | Upload an image **or** capture from webcam → predicted person label + confidence score |
| **Dark Mode** | Toggle between light and dark themes |
| **Auto-training** | Model is trained automatically on first run and saved to `model/saved_model.pkl` |
| **Error Handling** | Friendly JSON errors for invalid uploads and server issues |

---

## Folder Structure

```
for_ml_6th/
├── app.py                  # Flask backend – training, API endpoints
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── model/
│   └── saved_model.pkl     # Trained model pipeline (created at runtime)
├── templates/
│   └── index.html          # Main UI template
└── static/
    ├── style.css           # Custom styles (dark mode, animations)
    └── script.js           # Client-side JS (Fetch API, webcam, preview)
```

---

## Setup & Installation

### Prerequisites

* **Python 3.9+** (3.10 or 3.11 recommended)
* **pip** (comes with Python)
* An internet connection for the first run (to download the Olivetti Faces
  dataset from scikit-learn's servers)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Goutham-R1/for_ml_6th.git
cd for_ml_6th

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

On the **first run** the app will:

1. Download the Olivetti Faces dataset (~2 MB).
2. Train three models with cross-validation.
3. Save the best model to `model/saved_model.pkl`.

Subsequent runs load the saved model instantly.

Open your browser at **<http://localhost:5000>** to use the app.

---

## API Endpoints

### `GET /`

Renders the main HTML UI.

### `GET /api/models`

Returns model accuracy comparison.

```json
{
    "models": {
        "Gaussian Naive Bayes": 0.8275,
        "SVM": 0.955,
        "KNN": 0.935
    },
    "best_model": "SVM"
}
```

### `POST /api/predict`

Accepts a `multipart/form-data` upload with field name **`image`**.

**Success response:**

```json
{
    "predicted_label": 12,
    "confidence": 0.9731,
    "model_used": "SVM",
    "message": "Predicted as Person #12"
}
```

**Error response (example):**

```json
{
    "error": "Invalid file type. Allowed types: bmp, gif, jpeg, jpg, png, webp"
}
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **`ModuleNotFoundError`** | Make sure you ran `pip install -r requirements.txt` in the correct environment. |
| **Dataset download fails** | Check your internet connection. scikit-learn downloads the Olivetti dataset from the web on first use. |
| **Invalid image error** | Upload a standard image file (PNG, JPG, etc.). The image will be auto-resized to 64×64 grayscale. |
| **Port 5000 already in use** | Change the port in `app.py` (`app.run(port=5001)`) or stop the other process. |
| **Webcam not working** | Ensure you allow camera permissions in the browser prompt. HTTPS may be required on some browsers. |

---

## Dependencies

Listed in `requirements.txt`:

* **Flask** – web framework
* **scikit-learn** – ML models, PCA, cross-validation
* **NumPy** – numerical operations
* **Pillow** – image loading and conversion
* **opencv-python-headless** – image resizing

---

## License

This project is provided for educational purposes.  Feel free to use and modify
it for learning.
