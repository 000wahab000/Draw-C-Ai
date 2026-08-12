// ==========================================
// Draw Grid Capture Tool - Logic
// ==========================================
// This script handles the drawing on the canvas and converts the 
// visual drawing into a structured array (vector) of numbers.

const canvas = document.getElementById('draw-canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear-btn');
const saveBtn = document.getElementById('save-btn');
const classSelect = document.getElementById('class-select');
const statusDiv = document.getElementById('status');
const predictBtn = document.getElementById('predict-btn');
const predictionResult = document.getElementById('prediction-result');
const breakdownSection = document.getElementById('breakdown-section');
const breakdownBtn = document.getElementById('breakdown-btn');
const breakdownResult = document.getElementById('breakdown-result');
const modeSelect = document.getElementById('mode-select');

// Our logical grid size is 32x32.
// Since our canvas is visually 320x320, each logical pixel is visually 10x10.
const GRID_SIZE = 32;
const VISUAL_SIZE = 320;
const PIXEL_SCALE = VISUAL_SIZE / GRID_SIZE; // 10

// State variables to track if the user is currently drawing
let isDrawing = false;

// We use an internal 2D array to represent our 32x32 grid mathematically.
// 0 means blank/white, 1 means drawn/black.
// We initialize it with zeros.
let gridData = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0));

/**
 * Erases everything on the canvas and resets the logical grid array to 0.
 */
function clearCanvas() {
    ctx.clearRect(0, 0, VISUAL_SIZE, VISUAL_SIZE);

    // Reset our grid array
    for (let y = 0; y < GRID_SIZE; y++) {
        for (let x = 0; x < GRID_SIZE; x++) {
            gridData[y][x] = 0;
        }
    }
    statusDiv.innerText = "Canvas cleared.";
}

/**
 * Translates visual mouse coordinates (e.g. x: 145, y: 52) 
 * into grid coordinates (e.g. x: 14, y: 5)
 */
function getGridCoordinates(event) {
    const rect = canvas.getBoundingClientRect();

    // Compute actual scale: CSS may resize the canvas visually, so we can't
    // use the hardcoded PIXEL_SCALE. We derive the ratio from the real rect.
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const rawX = (event.clientX - rect.left) * scaleX;
    const rawY = (event.clientY - rect.top) * scaleY;

    const gridX = Math.floor(rawX / PIXEL_SCALE);
    const gridY = Math.floor(rawY / PIXEL_SCALE);

    return { gridX, gridY };
}

/**
 * Marks a grid cell as drawn (1), and renders a visual square on the canvas.
 */
function drawPixel(gridX, gridY) {
    // Prevent drawing outside the array boundaries
    if (gridX < 0 || gridX >= GRID_SIZE || gridY < 0 || gridY >= GRID_SIZE) {
        return;
    }

    // Set internal state to 1 (drawn)
    gridData[gridY][gridX] = 1;

    // Visually draw the black square on the canvas
    ctx.fillStyle = 'black';
    // fillRect(x, y, width, height)
    ctx.fillRect(gridX * PIXEL_SCALE, gridY * PIXEL_SCALE, PIXEL_SCALE, PIXEL_SCALE);

    // Optional: Draw a slightly lighter border so it looks like pixels
    // ctx.strokeStyle = '#333';
    // ctx.strokeRect(gridX * PIXEL_SCALE, gridY * PIXEL_SCALE, PIXEL_SCALE, PIXEL_SCALE);
}

// --- Event Listeners for Mouse ---

canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const { gridX, gridY } = getGridCoordinates(e);
    drawPixel(gridX, gridY);
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return; // Only draw if mouse is held down
    const { gridX, gridY } = getGridCoordinates(e);
    drawPixel(gridX, gridY);
});

canvas.addEventListener('mouseup', () => {
    isDrawing = false;
});

// Stop drawing when mouse leaves the canvas OR the entire window.
// This prevents a stuck "isDrawing" state if the user drags outside.
canvas.addEventListener('mouseleave', () => { isDrawing = false; });
window.addEventListener('mouseup', () => { isDrawing = false; });

// --- Actions ---

clearBtn.addEventListener('click', clearCanvas);

/**
 * Extracts the 2D grid array, flattens it into a 1D vector (1024 items),
 * and saves it to a JSON file.
 */
saveBtn.addEventListener('click', () => {
    // "Flattening" the array:
    // We convert [[0,1,0], [1,1,1], [0,0,0]] into [0,1,0,1,1,1,0,0,0]
    // A 32x32 grid becomes a 1024-length 1D array.
    const flattenedVector = gridData.flat();

    // We package the data together with its label so we know what class this is.
    const selectedClass = classSelect.value;

    const dataObj = {
        label: selectedClass,
        grid_size: GRID_SIZE,
        vector: flattenedVector // 1024 length array of 1s and 0s
    };

    // Convert the JavaScript object into a JSON string
    const jsonString = JSON.stringify(dataObj);

    // Create a virtual file to download
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    // Create a temporary anchor tag to trigger the browser download
    const a = document.createElement('a');
    a.href = url;
    // Naming format: className_timestamp.json
    a.download = `${selectedClass}_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();

    // Cleanup
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    // Let the user know
    statusDiv.innerText = `Saved ${selectedClass} vector successfully!`;

    // Clear canvas for the next drawing
    setTimeout(clearCanvas, 1000);
});


predictBtn.addEventListener('click', async () => {
    const flattenedVector = gridData.flat();
    const mode = modeSelect.value;
    
    predictionResult.innerText = mode === 'embedding_knn' ? 'Predicting (ResNet running...)' : 'Predicting...';
    predictionResult.style.color = 'blue';
    breakdownSection.style.display = 'none';
    breakdownResult.innerHTML = '';
    breakdownBtn.style.display = 'inline-block';
    
    try {
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                vector: flattenedVector,
                mode: mode,
                include_all_classes: false
            })
        });
        
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        
        if (data.error) {
            predictionResult.innerText = `Error: ${data.error}`;
            predictionResult.style.color = "red";
        } else {
            predictionResult.innerText = `[${mode}] ${data.label} (${(data.confidence * 100).toFixed(1)}%) in ${data.inference_ms}ms`;
            predictionResult.style.color = "green";
            breakdownSection.style.display = 'block';
        }
    } catch (error) {
        predictionResult.innerText = `Failed to connect to API: ${error.message}`;
        predictionResult.style.color = "red";
    }
});

breakdownBtn.addEventListener('click', async () => {
    const flattenedVector = gridData.flat();
    breakdownBtn.innerText = "Loading...";
    breakdownBtn.disabled = true;
    
    try {
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                vector: flattenedVector,
                mode: modeSelect.value,
                include_all_classes: true
            })
        });
        
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        
        let html = '';
        for (const [label, percent] of Object.entries(data.all_classes)) {
            html += `
                <div class="breakdown-row">
                    <div class="breakdown-label">${label}</div>
                    <div class="breakdown-bar-bg">
                        <div class="breakdown-bar-fill" style="width: ${percent}%;"></div>
                    </div>
                    <div class="breakdown-percent">${percent.toFixed(1)}%</div>
                </div>
            `;
        }
        breakdownResult.innerHTML = html;
        breakdownBtn.style.display = 'none'; // hide button after showing
        
    } catch (error) {
        breakdownResult.innerHTML = `<span style="color:red">Error: ${error.message}</span>`;
    } finally {
        breakdownBtn.innerText = "Show breakdown";
        breakdownBtn.disabled = false;
    }
});

// Initialize with a clean state
clearCanvas();
