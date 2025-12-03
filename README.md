# MotionGPT

Text-to-Motion and Motion-to-Text generation using a unified language model.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vamsisuhas/MotionGPT.git
cd MotionGPT
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install chumpy (required for SMPL rendering):
```bash
bash setup_chumpy.sh
```

## Download Models

1. Download T5 model:
```bash
# Models will be downloaded automatically on first run, or download manually:
# Place in deps/flan-t5-base/
```

2. Download Whisper model:
```bash
# Place in deps/whisper-large-v2/
```

3. Download SMPL models:
```bash
bash prepare/download_smpl_model.sh
```

4. Download checkpoint:
```bash
# Place checkpoint.tar in checkpoints/MotionGPT-base/
```

## Running the Application

### Local Development

Run the Gradio app:
```bash
python app.py
```

The app will be available at `http://localhost:7860`

### HuggingFace Spaces

The `hf_space/` directory contains the deployment-ready version for HuggingFace Spaces.

## Usage

1. **Text-to-Motion**: Enter a text description (e.g., "a person walks forward")
2. **Audio-to-Motion**: Upload an audio file (will be transcribed to text first)
3. **Motion-to-Text**: Upload a `.npy` motion file to get a text description

### Output

- **Fast Mode**: Skeleton visualization (quick preview)
- **Slow Mode**: SMPL 3D human body rendering (realistic, slower)

Both `.mp4` video and `.npy` motion files are available for download.

## Project Structure

```
MotionGPT/
├── app.py                 # Main Gradio application
├── mGPT/                  # Core model code
├── configs/               # Configuration files
├── checkpoints/           # Model checkpoints
├── deps/                  # External dependencies (T5, Whisper, SMPL)
├── hf_space/             # HuggingFace Spaces deployment files
└── requirements.txt       # Python dependencies
```

## Notes

- First run will download models automatically (if available)
- SMPL rendering requires chumpy and OpenGL libraries
- For headless environments (HuggingFace Spaces), OSMesa is used for rendering
