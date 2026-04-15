/**
 * Face Recognition System – Client-Side JavaScript
 * ==================================================
 * Handles:
 *   • Image file selection and preview
 *   • Form submission via Fetch API (no page reload)
 *   • Loading spinner during prediction
 *   • Result / error display
 *   • Fetching model accuracy comparison from /api/models
 *   • Dark-mode toggle
 *   • Webcam capture via getUserMedia
 */

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const uploadForm        = document.getElementById("uploadForm");
const imageInput        = document.getElementById("imageInput");
const predictBtn        = document.getElementById("predictBtn");
const previewContainer  = document.getElementById("previewContainer");
const imagePreview      = document.getElementById("imagePreview");

const spinner           = document.getElementById("spinner");
const resultSection     = document.getElementById("resultSection");
const resultLabel       = document.getElementById("resultLabel");
const resultConfidence  = document.getElementById("resultConfidence");
const resultModel       = document.getElementById("resultModel");
const errorSection      = document.getElementById("errorSection");
const errorMessage      = document.getElementById("errorMessage");
const placeholderSection = document.getElementById("placeholderSection");

const accuracySpinner   = document.getElementById("accuracySpinner");
const accuracySection   = document.getElementById("accuracySection");
const accuracyBody      = document.getElementById("accuracyBody");

const darkModeToggle    = document.getElementById("darkModeToggle");

const startWebcamBtn    = document.getElementById("startWebcamBtn");
const captureBtn        = document.getElementById("captureBtn");
const webcamVideo       = document.getElementById("webcamVideo");
const webcamCanvas      = document.getElementById("webcamCanvas");
const webcamFallback    = document.getElementById("webcamFallback");

// Webcam stream reference so we can stop it later.
let webcamStream = null;

// ---------------------------------------------------------------------------
// 1) Image preview on file selection
// ---------------------------------------------------------------------------
imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (file) {
        // Client-side type check
        if (!file.type.startsWith("image/")) {
            showError("Please select a valid image file.");
            imageInput.value = "";
            previewContainer.classList.add("d-none");
            predictBtn.disabled = true;
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove("d-none");
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    } else {
        previewContainer.classList.add("d-none");
        predictBtn.disabled = true;
    }
});

// ---------------------------------------------------------------------------
// 2) Form submission → /api/predict
// ---------------------------------------------------------------------------
uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!imageInput.files[0]) {
        showError("Please select an image first.");
        return;
    }

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    await sendPrediction(formData);
});

/**
 * Send a FormData object to /api/predict and handle the response.
 * @param {FormData} formData
 */
async function sendPrediction(formData) {
    // Show spinner, hide previous results
    showSpinner();

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            showError(data.error || "An unexpected error occurred.");
            return;
        }

        showResult(data);
    } catch (err) {
        showError("Network error – could not reach the server.");
    }
}

// ---------------------------------------------------------------------------
// 3) UI state helpers
// ---------------------------------------------------------------------------

function showSpinner() {
    spinner.classList.remove("d-none");
    resultSection.classList.add("d-none");
    errorSection.classList.add("d-none");
    placeholderSection.classList.add("d-none");
}

function showResult(data) {
    spinner.classList.add("d-none");
    errorSection.classList.add("d-none");
    placeholderSection.classList.add("d-none");
    resultSection.classList.remove("d-none");

    resultLabel.textContent = data.message || `Predicted: Person #${data.predicted_label}`;

    if (data.confidence !== null && data.confidence !== undefined) {
        resultConfidence.textContent =
            `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
    } else {
        resultConfidence.textContent = "Confidence: N/A";
    }

    resultModel.textContent = `Model used: ${data.model_used}`;
}

function showError(msg) {
    spinner.classList.add("d-none");
    resultSection.classList.add("d-none");
    placeholderSection.classList.add("d-none");
    errorSection.classList.remove("d-none");
    errorMessage.textContent = msg;
}

// ---------------------------------------------------------------------------
// 4) Fetch model accuracy comparison on page load
// ---------------------------------------------------------------------------

async function loadModelAccuracies() {
    try {
        const res = await fetch("/api/models");
        const data = await res.json();

        accuracySpinner.classList.add("d-none");
        accuracySection.classList.remove("d-none");

        // Build table rows
        accuracyBody.innerHTML = "";
        for (const [name, acc] of Object.entries(data.models)) {
            const isBest = name === data.best_model;
            const pct = (acc * 100).toFixed(2);

            // Pick a colour for the progress bar
            let barColor = "#6c757d";  // grey default
            if (acc >= 0.95)      barColor = "#198754";  // green
            else if (acc >= 0.85) barColor = "#0d6efd";  // blue
            else if (acc >= 0.70) barColor = "#ffc107";  // yellow
            else                  barColor = "#dc3545";  // red

            const row = document.createElement("tr");
            row.innerHTML = `
                <td class="fw-semibold">
                    ${name}
                    ${isBest ? '<span class="badge badge-best ms-1">Best</span>' : ""}
                </td>
                <td>${pct}%</td>
                <td style="min-width:120px">
                    <div class="accuracy-bar">
                        <div
                            class="accuracy-bar-fill"
                            style="width:${pct}%; background-color:${barColor}"
                        ></div>
                    </div>
                </td>
            `;
            accuracyBody.appendChild(row);
        }
    } catch {
        accuracySpinner.innerHTML =
            '<small class="text-danger">Failed to load model data.</small>';
    }
}

// Load accuracies as soon as the page is ready.
document.addEventListener("DOMContentLoaded", loadModelAccuracies);

// ---------------------------------------------------------------------------
// 5) Dark-mode toggle
// ---------------------------------------------------------------------------

function applyTheme(theme) {
    document.documentElement.setAttribute("data-bs-theme", theme);
    const icon = darkModeToggle.querySelector("i");
    if (theme === "dark") {
        icon.className = "bi bi-sun-fill";
    } else {
        icon.className = "bi bi-moon-stars-fill";
    }
}

// Restore saved preference (if any)
const savedTheme = localStorage.getItem("theme") || "light";
applyTheme(savedTheme);

darkModeToggle.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-bs-theme");
    const next = current === "dark" ? "light" : "dark";
    localStorage.setItem("theme", next);
    applyTheme(next);
});

// ---------------------------------------------------------------------------
// 6) Webcam capture
// ---------------------------------------------------------------------------

startWebcamBtn.addEventListener("click", async () => {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: 320, height: 240 },
        });
        webcamVideo.srcObject = webcamStream;
        webcamVideo.classList.remove("d-none");
        webcamFallback.classList.add("d-none");
        captureBtn.disabled = false;
        startWebcamBtn.textContent = "Webcam Active";
        startWebcamBtn.disabled = true;
    } catch {
        // Permission denied or no camera available
        webcamFallback.classList.remove("d-none");
        webcamVideo.classList.add("d-none");
        captureBtn.disabled = true;
    }
});

captureBtn.addEventListener("click", async () => {
    if (!webcamStream) return;

    // Draw current video frame to hidden canvas
    const ctx = webcamCanvas.getContext("2d");
    webcamCanvas.width = webcamVideo.videoWidth || 320;
    webcamCanvas.height = webcamVideo.videoHeight || 240;
    ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);

    // Convert canvas to Blob, then send as form data
    webcamCanvas.toBlob(async (blob) => {
        if (!blob) {
            showError("Failed to capture webcam frame.");
            return;
        }
        const formData = new FormData();
        formData.append("image", blob, "webcam_capture.png");
        await sendPrediction(formData);
    }, "image/png");
});
