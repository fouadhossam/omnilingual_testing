import os
import soundfile as sf
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
import threading
import threading as _threading

# Initialize fairseq2 thread-local storage early
try:
    # Import fairseq2 to trigger initialization
    import fairseq2
    # Try to access gang module to ensure thread-local is set up
    try:
        from fairseq2 import gang
        # Access thread-local to initialize it
        if not hasattr(gang._thread_local, 'current_gangs'):
            gang._thread_local.current_gangs = []
    except:
        pass
except:
    pass

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global state for progress tracking
transcription_progress = {}

# Global pipelines - one per model
pipelines = {}
model_loading = {}  # Track loading state per model
model_loading_lock = threading.Lock()  # Lock for model loading
DEFAULT_MODEL = 'omniASR_LLM_7B'
DEFAULT_DEVICE = 'cuda'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def chunk_audio(waveform, sr, chunk_sec, overlap_sec):
    chunk_size = int(chunk_sec * sr)
    overlap = int(overlap_sec * sr)
    step = chunk_size - overlap

    chunks = []
    for start in range(0, len(waveform), step):
        end = min(start + chunk_size, len(waveform))
        chunks.append(waveform[start:end])
        if end == len(waveform):
            break
    return chunks


def _init_thread_local():
    """Initialize fairseq2 thread-local storage in current thread"""
    try:
        # Try to access the thread-local storage to initialize it
        import fairseq2.gang as gang_module
        # Access the thread-local attribute to trigger initialization
        if not hasattr(gang_module._thread_local, 'current_gangs'):
            # Initialize it
            gang_module._thread_local.current_gangs = []
        # Try to create a dummy device to trigger full initialization
        try:
            import torch
            dummy_device = torch.device('cpu')
            # This might trigger initialization
        except:
            pass
    except (ImportError, AttributeError) as e:
        # If we can't initialize, continue anyway
        pass
    except Exception as e:
        # Other errors - log but continue
        print(f"Warning: Could not initialize thread-local storage: {e}")

def load_model(model, device):
    """Load model - initialize thread-local storage first"""
    global pipelines, model_loading
    model_key = f"{model}_{device}"
    
    # Initialize thread-local storage for this thread
    _init_thread_local()
    
    with model_loading_lock:
        if model_key in pipelines:
            return pipelines[model_key]
        
        if model_key in model_loading:
            return None  # Already loading
        
        model_loading[model_key] = True
    
    try:
        print(f"Loading model {model} on {device}...")
        pipeline = ASRInferencePipeline(model_card=model, device=device)
        print(f"Model {model} loaded successfully!")
        
        with model_loading_lock:
            pipelines[model_key] = pipeline
            del model_loading[model_key]
        
        return pipeline
    except Exception as e:
        print(f"Error loading model {model}: {str(e)}")
        import traceback
        traceback.print_exc()
        with model_loading_lock:
            if model_key in model_loading:
                del model_loading[model_key]
        raise e


def transcribe_long_audio(path, model, device, chunk_sec, overlap_sec, lang, task_id):
    global pipelines
    import time
    start_time = time.time()
    try:
        model_key = f"{model}_{device}"
        
        # Wait for model to be loaded if it's loading
        max_wait = 300  # 5 minutes max wait
        waited = 0
        while waited < max_wait:
            with model_loading_lock:
                if model_key in pipelines:
                    break
                if model_key not in model_loading:
                    # Model not loading and not loaded - try to load it
                    transcription_progress[task_id] = {
                        'status': 'error',
                        'error': f'Model {model} not loaded',
                        'message': f'Error: Model {model} not loaded. Please load it first.'
                    }
                    return
            time.sleep(0.5)
            waited += 0.5
        
        if model_key not in pipelines:
            transcription_progress[task_id] = {
                'status': 'error',
                'error': f'Model {model} loading timeout',
                'message': f'Error: Model {model} loading took too long'
            }
            return
        
        with model_loading_lock:
            pipeline = pipelines[model_key]
        
        # Load audio file
        audio, sr = sf.read(path)
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        duration = len(audio) / sr
        transcription_progress[task_id] = {
            'status': 'processing',
            'progress': 0,
            'total_chunks': 0,
            'current_chunk': 0,
            'duration': duration,
            'elapsed_time': 0,
            'chunk_text': '',
            'message': f'Loaded audio, duration: {duration:.2f}s'
        }

        # Chunk audio
        chunks = chunk_audio(audio, sr, chunk_sec, overlap_sec)
        transcription_progress[task_id]['total_chunks'] = len(chunks)

        final_text = ""
        last_tail = ""

        for i, chunk in enumerate(chunks):
            audio_input = [{"waveform": chunk, "sample_rate": sr}]

            out = pipeline.transcribe(
                audio_input,
                lang=[lang] if lang else None,
                batch_size=1
            )[0]

            # Simple stitching: avoid duplicating overlap words
            if last_tail and out.startswith(last_tail):
                out = out[len(last_tail):]

            final_text += " " + out.strip()

            # Store last few words of this chunk as tail
            last_tail = " ".join(out.split()[-10:])
            
            elapsed = time.time() - start_time
            transcription_progress[task_id]['current_chunk'] = i + 1
            transcription_progress[task_id]['progress'] = int((i + 1) / len(chunks) * 100)
            transcription_progress[task_id]['elapsed_time'] = round(elapsed, 1)
            transcription_progress[task_id]['chunk_text'] = final_text.strip()
            transcription_progress[task_id]['message'] = f'Transcribing chunk {i+1}/{len(chunks)}...'

        elapsed = time.time() - start_time
        transcription_progress[task_id] = {
            'status': 'completed',
            'progress': 100,
            'text': final_text.strip(),
            'elapsed_time': round(elapsed, 1),
            'message': f'Transcription completed in {elapsed:.1f}s!'
        }
    except Exception as e:
        transcription_progress[task_id] = {
            'status': 'error',
            'error': str(e),
            'message': f'Error: {str(e)}'
        }


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)


@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    # Get parameters
    model = request.form.get('model', DEFAULT_MODEL)
    device = request.form.get('device', DEFAULT_DEVICE)
    chunk_sec = float(request.form.get('chunk_sec', 30))
    overlap_sec = float(request.form.get('overlap_sec', 1.0))
    lang = request.form.get('lang', 'arz_Arab')

    # Validate chunk_sec
    if chunk_sec >= 40:
        return jsonify({'error': 'chunk_sec must be < 40 seconds'}), 400

    # Check for demo audio or uploaded file
    demo = request.form.get('demo')
    if demo:
        filepath = os.path.join('audios', demo)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Demo audio not found'}), 404
    else:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

    # Generate task ID
    import uuid
    task_id = str(uuid.uuid4())

    # Start transcription in background thread
    thread = threading.Thread(
        target=transcribe_long_audio,
        args=(filepath, model, device, chunk_sec, overlap_sec, lang, task_id)
    )
    thread.start()

    return jsonify({'task_id': task_id})


@app.route('/api/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    if task_id not in transcription_progress:
        return jsonify({'error': 'Task not found'}), 404

    return jsonify(transcription_progress[task_id])


@app.route('/api/demo_audios', methods=['GET'])
def get_demo_audios():
    demo_folder = 'audios'
    demos = []
    if os.path.exists(demo_folder):
        for f in os.listdir(demo_folder):
            if f.endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
                demos.append(f)
    return jsonify({'audios': sorted(demos)})


@app.route('/api/load_model', methods=['POST'])
def load_model_endpoint():
    """Load model synchronously in the request handler thread"""
    data = request.json
    model = data.get('model', DEFAULT_MODEL)
    device = data.get('device', DEFAULT_DEVICE)
    
    model_key = f"{model}_{device}"
    
    with model_loading_lock:
        if model_key in pipelines:
            return jsonify({'status': 'loaded', 'message': f'Model {model} already loaded'})
        
        if model_key in model_loading:
            return jsonify({'status': 'loading', 'message': f'Model {model} is loading...'})
    
    # Load model in background thread to avoid blocking
    def load_in_background():
        try:
            # Initialize thread-local in this thread
            _init_thread_local()
            load_model(model, device)
        except Exception as e:
            print(f"Background model loading failed: {e}")
            import traceback
            traceback.print_exc()
            with model_loading_lock:
                if model_key in model_loading:
                    del model_loading[model_key]
    
    # Start loading in a daemon thread
    thread = threading.Thread(target=load_in_background, daemon=True)
    thread.start()
    return jsonify({'status': 'loading', 'message': f'Loading model {model}...'})


@app.route('/api/model_status/<model>/<device>', methods=['GET'])
def get_model_status(model, device):
    model_key = f"{model}_{device}"
    
    with model_loading_lock:
        if model_key in pipelines:
            return jsonify({'status': 'loaded'})
        elif model_key in model_loading:
            return jsonify({'status': 'loading'})
        else:
            return jsonify({'status': 'not_loaded'})


if __name__ == '__main__':
    # Only load model in the actual server process, not in the Flask reloader parent process
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        print(f"Loading default ASR model {DEFAULT_MODEL}...")
        load_model(DEFAULT_MODEL, DEFAULT_DEVICE)
        print("Default model loaded successfully!")
    app.run(debug=True, host='0.0.0.0', port=5000)

