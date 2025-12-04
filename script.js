const API_BASE = 'http://localhost:5000/api';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const audioFileInput = document.getElementById('audioFile');
const fileName = document.getElementById('fileName');
const transcribeBtn = document.getElementById('transcribeBtn');
const resetBtn = document.getElementById('resetBtn');
const progressSection = document.getElementById('progressSection');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const resultSection = document.getElementById('resultSection');
const resultText = document.getElementById('resultText');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const copyBtn = document.getElementById('copyBtn');
const downloadBtn = document.getElementById('downloadBtn');
const modelSelect = document.getElementById('model');
const modelStatus = document.getElementById('modelStatus');

let currentTaskId = null;
let progressInterval = null;
let modelStatusInterval = null;
let currentModel = 'omniASR_LLM_7B';
let selectedDemo = null;

// Load demo audios
async function loadDemoAudios() {
    try {
        const response = await fetch(`${API_BASE}/demo_audios`);
        const data = await response.json();
        const demoContainer = document.getElementById('demoAudios');
        if (data.audios && data.audios.length > 0) {
            demoContainer.innerHTML = '<p class="demo-label">Demo Audios:</p><div class="demo-buttons"></div>';
            const buttonsContainer = demoContainer.querySelector('.demo-buttons');
            data.audios.forEach(audio => {
                const btn = document.createElement('button');
                btn.className = 'demo-btn';
                btn.textContent = audio;
                btn.onclick = () => selectDemo(audio);
                buttonsContainer.appendChild(btn);
            });
        }
    } catch (error) {
        console.error('Failed to load demo audios:', error);
    }
}

function selectDemo(audio) {
    selectedDemo = audio;
    audioFileInput.value = '';
    fileName.textContent = `Selected: ${audio}`;
    transcribeBtn.disabled = false;
    document.querySelectorAll('.demo-btn').forEach(btn => {
        btn.classList.toggle('active', btn.textContent === audio);
    });
    hideError();
}

// File Upload Handling
uploadArea.addEventListener('click', () => audioFileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

audioFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    selectedDemo = null;
    fileName.textContent = `Selected: ${file.name}`;
    transcribeBtn.disabled = false;
    document.querySelectorAll('.demo-btn').forEach(btn => btn.classList.remove('active'));
    hideError();
}

// Transcribe Button
transcribeBtn.addEventListener('click', async () => {
    if (!audioFileInput.files[0] && !selectedDemo) {
        showError('Please select an audio file or demo');
        return;
    }

    const selectedModel = document.getElementById('model').value;
    const device = 'cuda';
    
    // Check if model is loaded
    try {
        const statusResponse = await fetch(`${API_BASE}/model_status/${selectedModel}/${device}`);
        const statusData = await statusResponse.json();
        
        if (statusData.status !== 'loaded') {
            showError('Please wait for the model to finish loading');
            transcribeBtn.disabled = false;
            return;
        }
    } catch (error) {
        showError('Failed to check model status');
        transcribeBtn.disabled = false;
        return;
    }
    
    const formData = new FormData();
    if (selectedDemo) {
        formData.append('demo', selectedDemo);
    } else {
        formData.append('file', audioFileInput.files[0]);
    }
    formData.append('model', selectedModel);
    formData.append('device', device);
    formData.append('chunk_sec', '30');
    formData.append('overlap_sec', '1.0');
    formData.append('lang', document.getElementById('lang').value);

    try {
        transcribeBtn.disabled = true;
        hideError();
        showProgress();

        const response = await fetch(`${API_BASE}/transcribe`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to start transcription');
        }

        const data = await response.json();
        currentTaskId = data.task_id;

        // Start polling for progress
        startProgressPolling();
    } catch (error) {
        showError(error.message);
        transcribeBtn.disabled = false;
        hideProgress();
    }
});

// Progress Polling
function startProgressPolling() {
    if (progressInterval) {
        clearInterval(progressInterval);
    }

    progressInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/progress/${currentTaskId}`);
            if (!response.ok) {
                throw new Error('Failed to get progress');
            }

            const data = await response.json();

            if (data.status === 'completed') {
                clearInterval(progressInterval);
                showResult(data.text, data.elapsed_time);
                transcribeBtn.disabled = true;
                resetBtn.style.display = 'block';
            } else if (data.status === 'error') {
                clearInterval(progressInterval);
                showError(data.error || 'Transcription failed');
                transcribeBtn.disabled = false;
                hideProgress();
            } else if (data.status === 'processing') {
                updateProgress(data.progress, data.message, data.current_chunk, data.total_chunks, data.elapsed_time, data.chunk_text);
            }
        } catch (error) {
            clearInterval(progressInterval);
            showError('Failed to get progress: ' + error.message);
            transcribeBtn.disabled = false;
            hideProgress();
        }
    }, 1000); // Poll every second
}

function updateProgress(percent, message, currentChunk, totalChunks, elapsedTime, chunkText) {
    progressBar.style.width = `${percent}%`;
    let text = message;
    if (currentChunk && totalChunks) {
        text += ` (${currentChunk}/${totalChunks})`;
    }
    if (elapsedTime) {
        text += ` - ${elapsedTime}s`;
    }
    progressText.textContent = text;
    
    if (chunkText) {
        resultText.textContent = chunkText;
        resultSection.style.display = 'block';
    }
}

function showProgress() {
    progressSection.style.display = 'block';
    resultSection.style.display = 'none';
    progressBar.style.width = '0%';
    progressText.textContent = 'Initializing...';
}

function hideProgress() {
    progressSection.style.display = 'none';
}

function showResult(text, elapsedTime) {
    resultSection.style.display = 'block';
    resultText.textContent = text;
    if (elapsedTime) {
        progressText.textContent = `Completed in ${elapsedTime}s`;
    }
    hideProgress();
}

// Copy Button
copyBtn.addEventListener('click', () => {
    const text = resultText.textContent;
    navigator.clipboard.writeText(text).then(() => {
        const originalText = copyBtn.textContent;
        copyBtn.textContent = 'Copied!';
        setTimeout(() => {
            copyBtn.textContent = originalText;
        }, 2000);
    }).catch(err => {
        showError('Failed to copy to clipboard');
    });
});

// Download Button
downloadBtn.addEventListener('click', () => {
    const text = resultText.textContent;
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcription_${new Date().getTime()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});

// Reset Button
resetBtn.addEventListener('click', () => {
    audioFileInput.value = '';
    fileName.textContent = '';
    selectedDemo = null;
    document.querySelectorAll('.demo-btn').forEach(btn => btn.classList.remove('active'));
    transcribeBtn.disabled = true;
    resetBtn.style.display = 'none';
    resultSection.style.display = 'none';
    progressSection.style.display = 'none';
    hideError();
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    currentTaskId = null;
});

// Error Handling
function showError(message) {
    errorSection.style.display = 'block';
    errorMessage.textContent = message;
}

function hideError() {
    errorSection.style.display = 'none';
}

// Model Loading
modelSelect.addEventListener('change', async (e) => {
    const selectedModel = e.target.value;
    if (selectedModel === currentModel) return;
    
    currentModel = selectedModel;
    await loadModelIfNeeded(selectedModel);
});

async function loadModelIfNeeded(model) {
    const device = 'cuda';
    
    // Check current status
    try {
        const statusResponse = await fetch(`${API_BASE}/model_status/${model}/${device}`);
        const statusData = await statusResponse.json();
        
        if (statusData.status === 'loaded') {
            updateModelStatus('loaded', '✓ Loaded');
            return;
        }
        
        if (statusData.status === 'loading') {
            updateModelStatus('loading', '⏳ Loading...');
            startModelStatusPolling(model, device);
            return;
        }
        
        // Not loaded, start loading
        updateModelStatus('loading', '⏳ Loading...');
        const response = await fetch(`${API_BASE}/load_model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model, device })
        });
        
        const data = await response.json();
        if (data.status === 'loading') {
            startModelStatusPolling(model, device);
        } else if (data.status === 'error') {
            updateModelStatus('error', '✗ Error');
            showError(`Failed to load model: ${data.error}`);
        }
    } catch (error) {
        updateModelStatus('error', '✗ Error');
        showError(`Failed to check model status: ${error.message}`);
    }
}

function startModelStatusPolling(model, device) {
    if (modelStatusInterval) {
        clearInterval(modelStatusInterval);
    }
    
    modelStatusInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/model_status/${model}/${device}`);
            const data = await response.json();
            
            if (data.status === 'loaded') {
                clearInterval(modelStatusInterval);
                updateModelStatus('loaded', '✓ Loaded');
            } else if (data.status === 'loading') {
                updateModelStatus('loading', '⏳ Loading...');
            } else {
                clearInterval(modelStatusInterval);
                updateModelStatus('error', '✗ Not loaded');
            }
        } catch (error) {
            clearInterval(modelStatusInterval);
            updateModelStatus('error', '✗ Error');
        }
    }, 1000);
}

function updateModelStatus(status, text) {
    modelStatus.textContent = text;
    modelStatus.className = `model-status status-${status}`;
}

// Load default model and demo audios on page load
window.addEventListener('load', () => {
    loadModelIfNeeded(currentModel);
    loadDemoAudios();
});

