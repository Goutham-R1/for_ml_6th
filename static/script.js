/**
 * Image Recognition System – Client-Side JavaScript
 * ===================================================
 * Handles:
 *   • Image file selection and preview
 *   • Form submission via Fetch API (no page reload)
 *   • Loading spinner during analysis
 *   • Result / error display (face, object, not-found)
 *   • Fetching detection capabilities from /api/info
 *   • Dark-mode toggle
 *   • Webcam capture via getUserMedia
 */

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const uploadForm         = document.getElementById("uploadForm");
const imageInput         = document.getElementById("imageInput");
const predictBtn         = document.getElementById("predictBtn");
const previewContainer   = document.getElementById("previewContainer");
const imagePreview       = document.getElementById("imagePreview");

const spinner            = document.getElementById("spinner");
const resultSection      = document.getElementById("resultSection");
const resultAlert        = document.getElementById("resultAlert");
const resultLabel        = document.getElementById("resultLabel");
const resultConfidence   = document.getElementById("resultConfidence");
const resultDetails      = document.getElementById("resultDetails");
const errorSection       = document.getElementById("errorSection");
const errorMessage       = document.getElementById("errorMessage");
const placeholderSection = document.getElementById("placeholderSection");

const capabilitiesSpinner  = document.getElementById("capabilitiesSpinner");
const capabilitiesSection  = document.getElementById("capabilitiesSection");
const capabilitiesList     = document.getElementById("capabilitiesList");

const darkModeToggle     = document.getElementById("darkModeToggle");

const startWebcamBtn     = document.getElementById("startWebcamBtn");
const captureBtn         = document.getElementById("captureBtn");
const webcamVideo        = document.getElementById("webcamVideo");
const webcamCanvas       = document.getElementById("webcamCanvas");
const webcamFallback     = document.getElementById("webcamFallback");

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

/**
 * Choose an appropriate Bootstrap alert class based on the detection category.
 */
function alertClassForCategory(category) {
    switch (category) {
        case "human_face":    return "alert-success";
        case "cat":           return "alert-info";
        case "human_body":    return "alert-success";
        case "shape":         return "alert-primary";
        case "unknown_object": return "alert-warning";
        case "nothing":       return "alert-secondary";
        default:              return "alert-secondary";
    }
}

function showResult(data) {
    spinner.classList.add("d-none");
    errorSection.classList.add("d-none");
    placeholderSection.classList.add("d-none");
    resultSection.classList.remove("d-none");

    // Set alert colour based on category
    resultAlert.className = "alert " + alertClassForCategory(data.category);

    resultLabel.textContent = data.message || data.label;

    if (data.confidence !== null && data.confidence !== undefined && data.confidence > 0) {
        resultConfidence.textContent =
            `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    } else {
        resultConfidence.textContent = "";
    }

    resultDetails.textContent = data.details || "";
}

function showError(msg) {
    spinner.classList.add("d-none");
    resultSection.classList.add("d-none");
    placeholderSection.classList.add("d-none");
    errorSection.classList.remove("d-none");
    errorMessage.textContent = msg;
}

// ---------------------------------------------------------------------------
// 4) Fetch detection capabilities on page load
// ---------------------------------------------------------------------------

async function loadCapabilities() {
    try {
        const res = await fetch("/api/info");
        const data = await res.json();

        capabilitiesSpinner.classList.add("d-none");
        capabilitiesSection.classList.remove("d-none");

        capabilitiesList.innerHTML = "";
        for (const cap of data.capabilities) {
            const li = document.createElement("li");
            li.className = "list-group-item d-flex align-items-start gap-2";
            li.innerHTML = `
                <i class="bi ${cap.icon} fs-5 text-primary mt-1"></i>
                <div>
                    <strong>${cap.name}</strong>
                    <br/>
                    <small class="text-muted">${cap.description}</small>
                </div>
            `;
            capabilitiesList.appendChild(li);
        }
    } catch {
        capabilitiesSpinner.innerHTML =
            '<small class="text-danger">Failed to load capabilities.</small>';
    }
}

// Load capabilities as soon as the page is ready.
document.addEventListener("DOMContentLoaded", loadCapabilities);

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
