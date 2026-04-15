# Image Recognition System

A beginner-friendly web application that uses **OpenCV** and **Flask** to
detect and identify content in uploaded images.

---

## What It Does

Upload any image (or capture from webcam) and the system will tell you:

* **Human Face** → *"It's a Human Face!"*
* **Cat** → *"It's a Cat!"*
* **Human Body** → *"Detected a human body / person"*
* **Shapes** (circle, rectangle, triangle, …) → the shape name
* **Unknown Object** → *"Object Detected — could not identify the specific object"*
* **Blank / Nothing** → *"Not found anything"*

---

## Features

| Category | Details |
|----------|---------|
| **Face Detection** | OpenCV Haar Cascade (frontal face, with eye verification) |
| **Cat Detection** | OpenCV Haar Cascade (cat face) |
| **Body Detection** | OpenCV Haar Cascade (full body + upper body) |
| **Shape Detection** | Contour analysis for circles, rectangles, triangles, etc. |
| **Blank Detection** | Low pixel-variance check for empty images |
| **Dark Mode** | Toggle between light and dark themes |
| **Webcam** | Capture from webcam and analyse in real time |
| **Error Handling** | Friendly JSON errors for invalid uploads and server issues |

---

## Folder Structure

```
for_ml_6th/
├── app.py                  # Flask backend – detection logic, API endpoints
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── model/                  # (reserved for future use)
│   └── .gitkeep
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

Open your browser at **<http://localhost:5000>** to use the app.

---

## API Endpoints

### `GET /`

Renders the main HTML UI.

### `GET /api/info`

Returns the list of detection capabilities.

```json
{
    "capabilities": [
        { "name": "Human Face Detection", "description": "Detects human faces using Haar Cascade classifier", "icon": "bi-person-circle" },
        { "name": "Cat Detection",        "description": "Detects cat faces in images",                      "icon": "bi-heart-fill"    },
        { "name": "Human Body Detection",  "description": "Detects full or upper human body",                 "icon": "bi-person-standing"},
        { "name": "Shape Detection",       "description": "Identifies circles, rectangles, triangles, and more", "icon": "bi-pentagon"   },
        { "name": "Blank Detection",       "description": "Identifies blank or empty images",                "icon": "bi-x-circle"      }
    ]
}
```

### `POST /api/predict`

Accepts a `multipart/form-data` upload with field name **`image`**.

**Success response (face detected):**

```json
{
    "detected": true,
    "category": "human_face",
    "label": "Human Face",
    "message": "It's a Human Face! Detected a face in the image",
    "confidence": 0.85,
    "details": "Found 1 human face(s) and 2 eye(s)"
}
```

**Success response (nothing found):**

```json
{
    "detected": false,
    "category": "nothing",
    "label": "Not Found",
    "message": "Not found anything — no recognisable objects in the image",
    "confidence": 0.0,
    "details": "The image does not contain any clearly recognisable objects"
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
| **Invalid image error** | Upload a standard image file (PNG, JPG, etc.). |
| **Port 5000 already in use** | Change the port in `app.py` (`app.run(port=5001)`) or stop the other process. |
| **Webcam not working** | Ensure you allow camera permissions in the browser prompt. HTTPS may be required on some browsers. |

---

## Dependencies

Listed in `requirements.txt`:

* **Flask** – web framework
* **NumPy** – numerical operations
* **Pillow** – image loading and conversion
* **opencv-python-headless** – face/body/object detection via Haar Cascades

---

## License

This project is provided for educational purposes.  Feel free to use and modify
it for learning.
