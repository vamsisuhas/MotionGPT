# CRITICAL: Set up OSMesa and PyOpenGL BEFORE any OpenGL imports
# This must be done at the very top of the file, before any imports
import os
import sys
import subprocess

if os.getenv("SPACE_ID") is not None:  # HuggingFace Spaces
    print("🔧 Setting up OSMesa for HuggingFace Spaces...")
    # Set OSMesa platform BEFORE any OpenGL/pyglet imports
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    # Prevent pyglet from trying to use GLX (X11)
    os.environ['PYGLET_HIDE_WINDOW'] = '1'
    # Disable display (headless)
    os.environ['DISPLAY'] = ''
    
    # Uninstall PyOpenGL-accelerate (incompatible with OSMesa)
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", "-y", "PyOpenGL-accelerate"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass
    
    # Reinstall PyOpenGL to ensure it has GL_HALF_FLOAT and OSMesa support
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", "-y", "PyOpenGL"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--no-cache-dir", "PyOpenGL>=3.1.6"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ PyOpenGL reinstalled with OSMesa support")
    except Exception as e:
        print(f"⚠️  Could not reinstall PyOpenGL: {e}")
    
    # Patch GL_HALF_FLOAT if missing (must be done before any OpenGL imports)
    def patch_gl_half_float():
        try:
            from OpenGL.raw.GL import _types
            if not hasattr(_types, 'GL_HALF_FLOAT'):
                _types.GL_HALF_FLOAT = 0x140B  # GL_HALF_FLOAT constant value
                print("✅ Patched GL_HALF_FLOAT constant")
        except:
            pass
    
    # Register patch to run before OpenGL is imported
    import atexit
    atexit.register(patch_gl_half_float)
    
    # Also patch immediately if OpenGL is already imported somehow
    if 'OpenGL' in sys.modules:
        patch_gl_half_float()

import imageio
# Temporary workaround for Gradio import issue with huggingface_hub
try:
    import gradio as gr
except ImportError as e:
    if "HfFolder" in str(e):
        print("⚠️  Gradio import error due to huggingface_hub version mismatch.")
        print("   Attempting workaround...")
        # Try to patch huggingface_hub before importing gradio
        try:
            import huggingface_hub
            # Create a dummy HfFolder class if it doesn't exist
            if not hasattr(huggingface_hub, 'HfFolder'):
                class HfFolder:
                    @staticmethod
                    def save_token(token):
                        pass
                    @staticmethod
                    def get_token():
                        return None
                huggingface_hub.HfFolder = HfFolder
            import gradio as gr
            print("✅ Gradio imported with workaround")
        except Exception as patch_error:
            print(f"❌ Workaround failed: {patch_error}")
            print("   Please run: pip install 'huggingface_hub<0.20.0'")
            raise
    else:
        raise

# Patch Gradio to handle API schema generation errors
def patch_gradio_api_error():
    """Patch Gradio's schema parser to handle boolean additionalProperties"""
    try:
        import gradio_client.utils as gradio_client_utils
        
        # Patch the get_type function that's causing the error
        if hasattr(gradio_client_utils, 'get_type'):
            original_get_type = gradio_client_utils.get_type
            
            def safe_get_type(schema):
                # Handle case where schema is a boolean (True/False)
                if isinstance(schema, bool):
                    return "bool"
                # Handle case where schema is not a dict
                if not isinstance(schema, dict):
                    return "unknown"
                # Call original function for normal cases
                return original_get_type(schema)
            
            gradio_client_utils.get_type = safe_get_type
            
            # Also patch _json_schema_to_python_type to handle boolean additionalProperties
            if hasattr(gradio_client_utils, '_json_schema_to_python_type'):
                original_json_schema_to_python_type = gradio_client_utils._json_schema_to_python_type
                
                def safe_json_schema_to_python_type(schema, defs=None):
                    # Handle boolean additionalProperties
                    if isinstance(schema, bool):
                        return "bool"
                    if isinstance(schema, dict) and 'additionalProperties' in schema:
                        if isinstance(schema['additionalProperties'], bool):
                            # If additionalProperties is True/False, treat as dict/object
                            return "dict" if schema['additionalProperties'] else "dict"
                    try:
                        return original_json_schema_to_python_type(schema, defs)
                    except (TypeError, AttributeError) as e:
                        if "bool" in str(e) or "not iterable" in str(e) or "const" in str(e):
                            # Return a safe default
                            return "dict"
                        raise
                
                gradio_client_utils._json_schema_to_python_type = safe_json_schema_to_python_type
            
            print("✅ Patched Gradio schema parser")
        else:
            # Fallback: patch at Blocks level
            import gradio.blocks as gradio_blocks
            if hasattr(gradio_blocks, 'Blocks'):
                original_get_api_info = gradio_blocks.Blocks.get_api_info
                
                def safe_get_api_info(self):
                    try:
                        return original_get_api_info(self)
                    except (TypeError, AttributeError) as e:
                        if "bool" in str(e) or "not iterable" in str(e) or "const" in str(e):
                            print("⚠️  API schema generation error caught, returning empty API info")
                            return {}
                        raise
                
                gradio_blocks.Blocks.get_api_info = safe_get_api_info
                print("✅ Patched Gradio Blocks.get_api_info (fallback)")
    except Exception as e:
        print(f"⚠️  Could not patch Gradio API: {e}")
        import traceback
        traceback.print_exc()

patch_gradio_api_error()
import random
import torch
import time
import threading
import cv2
import os
import shutil
import subprocess
import sys
import numpy as np

import pytorch_lightning as pl
from moviepy import VideoFileClip
from pathlib import Path

# Fix chumpy compatibility with NumPy 1.23+ (MUST be before chumpy import)
# Patch at module level for 'from numpy import bool' to work
# Always set these attributes (they may not exist in newer numpy versions)
import numpy
numpy.bool = numpy.bool_
numpy.int = numpy.int_
numpy.float = numpy.float_
numpy.complex = numpy.complex_
numpy.object = numpy.object_
numpy.unicode = numpy.str_
numpy.str = numpy.str_
# Also patch np alias for consistency
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.complex = np.complex_
np.object = np.object_
np.unicode = np.str_
np.str = np.str_

# Install and import chumpy (REQUIRED for SMPL rendering - slow mode)
chumpy = None
try:
    import chumpy
    print("✅ chumpy imported successfully")
except (ImportError, AttributeError, ModuleNotFoundError) as e:
    print("📦 chumpy not found. Installing chumpy (REQUIRED for slow mode/SMPL rendering)...")
    try:
        # Install with verbose output to see any errors
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-build-isolation", "chumpy"])
        # Re-import after installation
        import importlib
        if 'chumpy' in sys.modules:
            del sys.modules['chumpy']
        import chumpy
        print("✅ chumpy installed and imported successfully")
    except Exception as install_error:
        print(f"❌ CRITICAL: Failed to install/import chumpy: {install_error}")
        print("   Slow mode (SMPL rendering) will NOT work without chumpy.")
        raise RuntimeError(f"chumpy is required for slow mode but failed to install/import: {install_error}")

from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.config import parse_args
from scipy.spatial.transform import Rotation as RRR
import mGPT.render.matplot.plot_3d_global as plot_3d
from mGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
# Import SMPLRender (REQUIRED for slow mode)
if chumpy is None:
    raise RuntimeError("chumpy must be imported before SMPLRender")

# Patch GL_HALF_FLOAT before importing pyrender (which imports OpenGL)
if os.getenv("SPACE_ID") is not None:
    try:
        # Import OpenGL types and patch if needed
        from OpenGL.raw.GL import _types
        if not hasattr(_types, 'GL_HALF_FLOAT'):
            _types.GL_HALF_FLOAT = 0x140B
            print("✅ Patched GL_HALF_FLOAT before pyrender import")
    except:
        pass

try:
    from mGPT.render.pyrender.smpl_render import SMPLRender
    print("✅ SMPLRender imported successfully")
except Exception as e:
    print(f"❌ CRITICAL: Could not import SMPLRender: {e}")
    raise RuntimeError(f"SMPLRender is required for slow mode but failed to import: {e}")
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# OpenGL platform is set at the top of the file (line 5) for HuggingFace Spaces
# For local environments, it will use the default (EGL or GLX)

# Download models from HuggingFace Hub if not present locally
def download_model_if_needed(repo_id, local_path, repo_type="model"):
    """Download model from HuggingFace Hub if local path doesn't exist"""
    if os.path.exists(local_path):
        return
    
    print(f"📥 Downloading {repo_id} to {local_path}...")
    try:
        from huggingface_hub import snapshot_download
        hf_username = os.getenv("HF_USERNAME", "vsadhu1")
        full_repo_id = repo_id if "/" in repo_id else f"{hf_username}/{repo_id}"
        
        # For checkpoint file, download to parent directory
        if local_path.endswith(".tar"):
            target_dir = Path(local_path).parent
            target_dir.mkdir(parents=True, exist_ok=True)
            # Download to temp location first
            temp_dir = snapshot_download(
                repo_id=full_repo_id,
                repo_type=repo_type,
                local_dir=str(target_dir / "temp"),
                local_dir_use_symlinks=False
            )
            # Find the .tar file and move it
            for file in Path(temp_dir).rglob("*.tar"):
                shutil.move(str(file), local_path)
                print(f"  ✅ Downloaded checkpoint to {local_path}")
                shutil.rmtree(Path(temp_dir).parent / "temp", ignore_errors=True)
                return
        else:
            # For directories, download directly to the target path
            target_path = Path(local_path).resolve()  # Get absolute path
            target_path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=full_repo_id,
                repo_type=repo_type,
                local_dir=str(target_path),
            )
            print(f"  ✅ Downloaded to {target_path}")
    except Exception as e:
        print(f"  ⚠️  Failed to download {repo_id}: {e}")
        print(f"  Please ensure models are available or upload them first")

# Download models if needed (for HuggingFace Spaces deployment)
is_hf_space = os.getenv("SPACE_ID") is not None
if is_hf_space:
    # Uninstall PyOpenGL-accelerate if present (incompatible with OSMesa)
    # This should be handled by packages.txt installing OSMesa, but ensure accelerate is not installed
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", "-y", "PyOpenGL-accelerate"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass  # Not installed, that's fine
    
    # PyOpenGL setup is already done at the top of the file
    # Just ensure GL_HALF_FLOAT is patched if needed
    try:
        from OpenGL.raw.GL import _types
        if not hasattr(_types, 'GL_HALF_FLOAT'):
            _types.GL_HALF_FLOAT = 0x140B
            print("✅ Patched GL_HALF_FLOAT constant")
    except:
        pass
    
    hf_username = os.getenv("HF_USERNAME", "vsadhu1")
    print("🌐 HuggingFace Spaces detected - downloading models...")
    
    # Download checkpoint
    download_model_if_needed(
        f"{hf_username}/MotionGPT-checkpoint",
        "checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar"
    )
    
    # Download T5 model
    download_model_if_needed(
        f"{hf_username}/MotionGPT-t5-base",
        "deps/flan-t5-base"
    )
    
    # Download Whisper model
    download_model_if_needed(
        f"{hf_username}/MotionGPT-whisper-large-v2",
        "deps/whisper-large-v2"
    )
    
    # Download SMPL models
    download_model_if_needed(
        f"{hf_username}/MotionGPT-smpl-models",
        "deps/smpl_models"
    )

# Load model
cfg = parse_args(phase="webui")  # parse config file

# Validate slow mode dependencies
def validate_slow_mode():
    """Validate that all dependencies for slow mode (SMPL rendering) are available"""
    errors = []
    
    if chumpy is None:
        errors.append("❌ chumpy is not imported")
    else:
        print("✅ chumpy is available")
    
    if SMPLRender is None:
        errors.append("❌ SMPLRender is not imported")
    else:
        print("✅ SMPLRender is available")
    
    smpl_model_path = cfg.RENDER.SMPL_MODEL_PATH
    # Check if path exists, also check parent directory (for hf_space/)
    app_dir = Path(__file__).parent.absolute()
    if not os.path.exists(smpl_model_path):
        # Try parent directory
        clean_path = smpl_model_path[2:] if smpl_model_path.startswith('./') else smpl_model_path
        parent_path = (app_dir.parent / clean_path).resolve()
        if parent_path.exists():
            print(f"✅ SMPL model path exists (in parent): {parent_path}")
        else:
            errors.append(f"❌ SMPL model path does not exist: {smpl_model_path}")
            errors.append(f"   Absolute path: {os.path.abspath(smpl_model_path)}")
            errors.append(f"   Parent path: {parent_path}")
    else:
        print(f"✅ SMPL model path exists: {smpl_model_path}")
    
    if errors:
        print("\n⚠️  SLOW MODE VALIDATION FAILED:")
        for error in errors:
            print(f"   {error}")
        print("\n   Slow mode will fail with clear error messages when attempted.")
        print("   Fast mode will continue to work normally.\n")
    else:
        print("✅ All slow mode dependencies validated successfully\n")

validate_slow_mode()

# Fix relative paths in config to absolute paths (required for transformers)
# Use app.py's directory as base for resolving relative paths
app_dir = Path(__file__).parent.absolute()
print(f"🔍 App directory: {app_dir}")

if hasattr(cfg, 'model') and hasattr(cfg.model, 'params') and hasattr(cfg.model.params, 'lm'):
    # lm is a DictConfig with 'target' and 'params' keys
    if hasattr(cfg.model.params.lm, 'params') and hasattr(cfg.model.params.lm.params, 'model_path'):
        model_path = cfg.model.params.lm.params.model_path
        print(f"🔍 Original model_path: {model_path}")
        # If it's a relative path (starts with ./) or local path (not a HF repo ID format)
        # Resolve to absolute path using app.py's directory as base
        if model_path.startswith('./') or (not os.path.isabs(model_path) and '/' in model_path and not model_path.count('/') == 1 and not model_path.startswith('google/') and not model_path.startswith('openai/')):
            # Remove ./ prefix if present
            clean_path = model_path[2:] if model_path.startswith('./') else model_path
            abs_path = (app_dir / clean_path).resolve()
            print(f"🔍 Checking: {abs_path} (exists: {abs_path.exists()})")
            # Update if the path exists (local file)
            if abs_path.exists():
                # Direct assignment works with OmegaConf DictConfig
                cfg.model.params.lm.params.model_path = str(abs_path)
                print(f"📝 Resolved model_path: {model_path} -> {abs_path}")
            else:
                # Try parent directory (in case running from hf_space/)
                parent_abs_path = (app_dir.parent / clean_path).resolve()
                print(f"🔍 Checking parent: {parent_abs_path} (exists: {parent_abs_path.exists()})")
                if parent_abs_path.exists():
                    # Direct assignment works with OmegaConf DictConfig
                    cfg.model.params.lm.params.model_path = str(parent_abs_path)
                    print(f"📝 Resolved model_path: {model_path} -> {parent_abs_path} (from parent directory)")
                else:
                    print(f"⚠️  Model path {model_path} not found at {abs_path} or {parent_abs_path}. Keeping original path.")
        else:
            print(f"⚠️  Model path {model_path} doesn't match relative path pattern. Skipping resolution.")

# Fix whisper_path similarly
if hasattr(cfg, 'model') and hasattr(cfg.model, 'whisper_path'):
    whisper_path = cfg.model.whisper_path
    # Check if it's a relative path (not absolute, contains /, and not a HF repo ID like google/flan-t5-base)
    # HF repo IDs have exactly 1 / and don't start with ./ or common local prefixes
    is_local_path = (not os.path.isabs(whisper_path) and '/' in whisper_path and 
                     (whisper_path.startswith('./') or 
                      whisper_path.startswith('deps/') or 
                      whisper_path.count('/') > 1 or
                      (whisper_path.count('/') == 1 and not whisper_path.startswith('google/') and not whisper_path.startswith('openai/'))))
    if is_local_path:
        clean_path = whisper_path[2:] if whisper_path.startswith('./') else whisper_path
        abs_path = (app_dir / clean_path).resolve()
        if abs_path.exists():
            cfg.model.whisper_path = str(abs_path)
            print(f"📝 Resolved whisper_path: {whisper_path} -> {abs_path}")
        else:
            parent_abs_path = (app_dir.parent / clean_path).resolve()
            if parent_abs_path.exists():
                cfg.model.whisper_path = str(parent_abs_path)
                print(f"📝 Resolved whisper_path: {whisper_path} -> {parent_abs_path} (from parent directory)")
            else:
                print(f"⚠️  Whisper path {whisper_path} not found at {abs_path} or {parent_abs_path}. Keeping original path.")

# Fix checkpoint path similarly
if hasattr(cfg, 'TEST') and hasattr(cfg.TEST, 'CHECKPOINTS'):
    checkpoint_path = cfg.TEST.CHECKPOINTS
    if checkpoint_path and (checkpoint_path.startswith('./') or (not os.path.isabs(checkpoint_path) and '/' in checkpoint_path)):
        clean_path = checkpoint_path[2:] if checkpoint_path.startswith('./') else checkpoint_path
        abs_path = (app_dir / clean_path).resolve()
        if abs_path.exists():
            cfg.TEST.CHECKPOINTS = str(abs_path)
            print(f"📝 Resolved checkpoint_path: {checkpoint_path} -> {abs_path}")
        else:
            parent_abs_path = (app_dir.parent / clean_path).resolve()
            if parent_abs_path.exists():
                cfg.TEST.CHECKPOINTS = str(parent_abs_path)
                print(f"📝 Resolved checkpoint_path: {checkpoint_path} -> {parent_abs_path} (from parent directory)")

# Fix SMPL model path similarly
if hasattr(cfg, 'RENDER') and hasattr(cfg.RENDER, 'SMPL_MODEL_PATH'):
    smpl_path = cfg.RENDER.SMPL_MODEL_PATH
    if smpl_path and (smpl_path.startswith('./') or (not os.path.isabs(smpl_path) and '/' in smpl_path)):
        clean_path = smpl_path[2:] if smpl_path.startswith('./') else smpl_path
        abs_path = (app_dir / clean_path).resolve()
        if abs_path.exists():
            cfg.RENDER.SMPL_MODEL_PATH = str(abs_path)
            print(f"📝 Resolved SMPL_MODEL_PATH: {smpl_path} -> {abs_path}")
        else:
            parent_abs_path = (app_dir.parent / clean_path).resolve()
            if parent_abs_path.exists():
                cfg.RENDER.SMPL_MODEL_PATH = str(parent_abs_path)
                print(f"📝 Resolved SMPL_MODEL_PATH: {smpl_path} -> {parent_abs_path} (from parent directory)")

cfg.FOLDER = 'cache'
output_dir = Path("assets")
output_dir.mkdir(parents=True, exist_ok=True)
pl.seed_everything(cfg.SEED_VALUE)
if cfg.ACCELERATOR == "gpu":
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
datamodule = build_data(cfg, phase="test")
model = build_model(cfg, datamodule)
state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
model.load_state_dict(state_dict)
model.to(device)

audio_processor = WhisperProcessor.from_pretrained(cfg.model.whisper_path)
audio_model = WhisperForConditionalGeneration.from_pretrained(cfg.model.whisper_path).to(device)
forced_decoder_ids = audio_processor.get_decoder_prompt_ids(language="zh", task="translate")
forced_decoder_ids_zh = audio_processor.get_decoder_prompt_ids(language="zh", task="translate")
forced_decoder_ids_en = audio_processor.get_decoder_prompt_ids(language="en", task="translate")

def ensure_absolute_video_path(video_path: str) -> str:
    """Convert a relative video path to an absolute path for Gradio uploads."""
    if isinstance(video_path, str) and not os.path.isabs(video_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        video_path = os.path.join(base_dir, video_path)
    return video_path


def create_bot_message(content):
    """Create an assistant message for Gradio's messages format."""
    return {"role": "assistant", "content": content}


def create_user_message(content):
    """Create a user message for Gradio's messages format."""
    return {"role": "user", "content": content}

def create_video_message(video_path):
    """Create a video message for Gradio 5.x chatbot"""
    abs_path = ensure_absolute_video_path(video_path)
    return create_bot_message({"path": abs_path, "mime_type": "video/mp4"})

def create_example_video(video_path):
    """Create a video reference for examples"""
    return create_video_message(video_path)

def create_download_links(video_path, motion_path, video_fname, motion_fname):
    """Create download links for video and motion files"""
    import os
    # Get absolute paths for downloads
    abs_video_path = os.path.abspath(video_path)
    abs_motion_path = os.path.abspath(motion_path)
    
    text = f"""**Generated Files:**
- **Video:** `{video_fname}` → saved to `{video_path}`
- **Motion Data:** `{motion_fname}` → saved to `{motion_path}`

**To download:** Right-click on the video above and select "Save video as..." or access files directly from the paths shown above."""
    return create_bot_message(text)


def motion_token_to_string(motion_token, lengths, codebook_size=512):
    motion_string = []
    for i in range(motion_token.shape[0]):
        motion_i = motion_token[i].cpu(
        ) if motion_token.device.type == 'cuda' else motion_token[i]
        motion_list = motion_i.tolist()[:lengths[i]]
        motion_string.append(
            (f'<motion_id_{codebook_size}>' +
             ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
             f'<motion_id_{codebook_size + 1}>'))
    return motion_string


def render_motion(data, feats, method='fast'):
    fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
        time.time())) + str(np.random.randint(10000, 99999))
    video_fname = fname + '.mp4'
    feats_fname = fname + '.npy'
    output_npy_path = os.path.join(output_dir, feats_fname)
    output_mp4_path = os.path.join(output_dir, video_fname)
    np.save(output_npy_path, feats)

    if method == 'slow':
        # Validate slow mode dependencies
        if SMPLRender is None:
            raise RuntimeError("SMPLRender is not available. Cannot use slow mode.")
        
        smpl_model_path = cfg.RENDER.SMPL_MODEL_PATH
        if not os.path.exists(smpl_model_path):
            raise FileNotFoundError(
                f"SMPL model path does not exist: {smpl_model_path}\n"
                f"Slow mode requires SMPL models to be downloaded. "
                f"Expected path: {os.path.abspath(smpl_model_path)}"
            )
        
        # Perform slow mode rendering (SMPL)
        if len(data.shape) == 4:
            data = data[0]
        data = data - data[0, 0]
        pose_generator = HybrIKJointsToRotmat()
        pose = pose_generator(data)
        pose = np.concatenate([
            pose,
            np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)
        ], 1)
        shape = [768, 768]
        # Force CPU for SMPL rendering to avoid CUDA compatibility issues
        # (PyTorch may not support older GPUs like V100 with CUDA 7.0)
        original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA to force CPU
        
        # Try OSMesa for headless environments if EGL fails
        original_pyopengl = os.environ.get('PYOPENGL_PLATFORM', None)
        
        try:
            render = SMPLRender(smpl_model_path)
            # Ensure it's using CPU (in case CUDA_VISIBLE_DEVICES didn't work)
            if render.device.type == 'cuda':
                print("⚠️  Renderer is using CUDA, forcing to CPU for compatibility...")
                render.device = torch.device("cpu")
                render.smpl = render.smpl.cpu()
        except (ImportError, OSError) as e:
            if "EGL" in str(e) or "egl" in str(e).lower():
                # EGL failed, try OSMesa (software rendering for headless)
                print("⚠️  EGL not available, trying OSMesa (software rendering)...")
                os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
                try:
                    render = SMPLRender(smpl_model_path)
                    if render.device.type == 'cuda':
                        render.device = torch.device("cpu")
                        render.smpl = render.smpl.cpu()
                except Exception as osmesa_error:
                    print(f"❌ OSMesa also failed: {osmesa_error}")
                    raise RuntimeError(
                        "Slow mode (SMPL rendering) requires OpenGL/EGL or OSMesa. "
                        "Neither is available in this environment. "
                        "Please use fast mode instead, or install OpenGL libraries."
                    )
            else:
                raise
        finally:
            # Restore original settings
            if original_cuda_visible is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
            else:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            if original_pyopengl is not None:
                os.environ['PYOPENGL_PLATFORM'] = original_pyopengl

        r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
        pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])
        vid = []
        aroot = data[[0], 0]
        aroot[:, 1] = -aroot[:, 1]
        params = dict(pred_shape=np.zeros([1, 10]),
                      pred_root=aroot,
                      pred_pose=pose)
        try:
            render.init_renderer([shape[0], shape[1], 3], params)
        except (ImportError, OSError, RuntimeError, AttributeError) as e:
            error_str = str(e)
            if any(x in error_str for x in ["EGL", "egl", "OpenGL", "OSMesa", "osmesa", "GLXPlatform"]):
                # OpenGL/EGL/OSMesa error - try to fix by reinstalling/reinitializing
                if is_hf_space:
                    # In HuggingFace Spaces, OSMesa should be installed via packages.txt
                    # If we get here, it means OSMesa is not properly installed
                    raise RuntimeError(
                        "Slow mode (SMPL rendering) requires OSMesa libraries. "
                        "Please ensure packages.txt includes: libosmesa6-dev libgl1 libglx-mesa0. "
                        f"Error: {error_str}"
                    )
                else:
                    raise RuntimeError(
                        f"Slow mode (SMPL rendering) failed: {error_str}. "
                        "Please check that OpenGL/EGL libraries are installed."
                    )
            else:
                raise
        
        for i in range(data.shape[0]):
            try:
                renderImg = render.render(i)
                vid.append(renderImg)
            except (TypeError, AttributeError) as render_error:
                # PyOpenGL-accelerate causes TypeError during rendering
                if "NoneType" in str(render_error) or "zeros()" in str(render_error):
                    print(f"⚠️  Rendering error (PyOpenGL-accelerate): {render_error}")
                    print("   Uninstalling PyOpenGL-accelerate and retrying...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "uninstall", "-y", "PyOpenGL-accelerate"
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    # Clear module cache
                    modules_to_clear = [m for m in sys.modules.keys() if 'OpenGL' in m or 'pyrender' in m]
                    for m in modules_to_clear:
                        del sys.modules[m]
                    # Recreate renderer
                    render = SMPLRender(smpl_model_path)
                    if render.device.type == 'cuda':
                        render.device = torch.device("cpu")
                        render.smpl = render.smpl.cpu()
                    render.init_renderer([shape[0], shape[1], 3], params)
                    # Retry rendering
                    renderImg = render.render(i)
                    vid.append(renderImg)
                else:
                    raise

        out = np.stack(vid, axis=0)
        output_gif_path = output_mp4_path[:-4] + '.gif'
        imageio.mimwrite(output_gif_path, out, duration=50)
        out_video = VideoFileClip(output_gif_path)
        out_video.write_videofile(output_mp4_path)
        del out, render

    elif method == 'fast':
        output_gif_path = output_mp4_path[:-4] + '.gif'
        if len(data.shape) == 3:
            data = data[None]
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        pose_vis = plot_3d.draw_to_batch(data, [''], [output_gif_path])
        out_video = VideoFileClip(output_gif_path)
        out_video.write_videofile(output_mp4_path)
        del pose_vis
    else:
        raise ValueError(f"Unknown rendering method: {method}. Must be 'slow' or 'fast'.")

    return output_mp4_path, video_fname, output_npy_path, feats_fname


def load_motion(motion_uploaded, method):
    file = motion_uploaded['file']

    feats = torch.tensor(np.load(file), device=model.device)
    if len(feats.shape) == 2:
        feats = feats[None]
    # feats = model.datamodule.normalize(feats)

    # Motion tokens
    motion_lengths = feats.shape[0]
    motion_token, _ = model.vae.encode(feats)

    motion_token_string = model.lm.motion_token_to_string(
        motion_token, [motion_token.shape[1]])[0]
    motion_token_length = motion_token.shape[1]

    # Motion rendered
    joints = model.datamodule.feats2joints(feats.cpu()).cpu().numpy()
    output_mp4_path, video_fname, output_npy_path, joints_fname = render_motion(
        joints,
        feats.to('cpu').numpy(), method)

    motion_uploaded.update({
        "feats": feats,
        "joints": joints,
        "motion_video": output_mp4_path,
        "motion_video_fname": video_fname,
        "motion_joints": output_npy_path,
        "motion_joints_fname": joints_fname,
        "motion_lengths": motion_lengths,
        "motion_token": motion_token,
        "motion_token_string": motion_token_string,
        "motion_token_length": motion_token_length,
    })

    return motion_uploaded


def add_text(history, text, motion_uploaded, data_stored, method):
    data_stored = data_stored + [{'user_input': text}]

    history = history + [create_user_message(text)]
    if 'file' in motion_uploaded.keys():
        motion_uploaded = load_motion(motion_uploaded, method)
        output_mp4_path = motion_uploaded['motion_video']
        video_fname = motion_uploaded['motion_video_fname']
        output_npy_path = motion_uploaded['motion_joints']
        joints_fname = motion_uploaded['motion_joints_fname']
        
        # Add video using Gradio 5.x messages format
        video_msg = create_video_message(output_mp4_path)
        history = history + [video_msg]

    return history, gr.update(value="",
                              interactive=False), motion_uploaded, data_stored


def add_audio(history, audio_path, data_stored, language='en'):
    audio, sampling_rate = librosa.load(audio_path, sr=16000)
    input_features = audio_processor(
        audio, sampling_rate, return_tensors="pt"
    ).input_features  # whisper training sampling rate, do not modify
    input_features = torch.Tensor(input_features).to(device)

    if language == 'English':
        forced_decoder_ids = forced_decoder_ids_en
    else:
        forced_decoder_ids = forced_decoder_ids_zh
    predicted_ids = audio_model.generate(input_features,
                                         forced_decoder_ids=forced_decoder_ids)
    text_input = audio_processor.batch_decode(predicted_ids,
                                              skip_special_tokens=True)
    text_input = str(text_input).strip('[]"')
    data_stored = data_stored + [{'user_input': text_input}]
    gr.update(value=data_stored, interactive=False)
    history = history + [create_user_message(text_input)]

    return history, data_stored


def add_file(history, file, txt, motion_uploaded):
    motion_uploaded['file'] = file.name
    txt = txt.replace(" <Motion_Placeholder>", "") + " <Motion_Placeholder>"
    return history, gr.update(value=txt, interactive=True), motion_uploaded


def bot(history, motion_uploaded, data_stored, method):

    motion_length, motion_token_string = motion_uploaded[
        "motion_lengths"], motion_uploaded["motion_token_string"]

    input = data_stored[-1]['user_input']
    prompt = model.lm.placeholder_fulfill(input, motion_length,
                                          motion_token_string, "")
    data_stored[-1]['model_input'] = prompt
    batch = {
        "length": [motion_length],
        "text": [prompt],
    }

    outputs = model(batch, task="t2m")
    out_feats = outputs["feats"][0]
    out_lengths = outputs["length"][0]
    out_joints = outputs["joints"][:out_lengths].detach().cpu().numpy()
    out_texts = outputs["texts"][0]
    output_mp4_path, video_fname, output_npy_path, joints_fname = render_motion(
        out_joints,
        out_feats.to('cpu').numpy(), method)

    motion_uploaded = {
        "feats": None,
        "joints": None,
        "motion_video": None,
        "motion_lengths": 0,
        "motion_token": None,
        "motion_token_string": '',
        "motion_token_length": 0,
    }

    data_stored[-1]['model_output'] = {
        "feats": out_feats,
        "joints": out_joints,
        "length": out_lengths,
        "texts": out_texts,
        "motion_video": output_mp4_path,
        "motion_video_fname": video_fname,
        "motion_joints": output_npy_path,
        "motion_joints_fname": joints_fname,
    }

    if '<Motion_Placeholder>' == out_texts:
        response = f"Generated motion video: {video_fname}"
        is_motion_generation = True
    elif '<Motion_Placeholder>' in out_texts:
        response = f"{out_texts.split('<Motion_Placeholder>')[0]} Generated motion video: {video_fname} {out_texts.split('<Motion_Placeholder>')[1]}"
        is_motion_generation = True
    else:
        # This is motion-to-text task, only show text description
        response = f"{out_texts}"
        is_motion_generation = False

    # Add bot response - animate text character by character
    bot_response_msg = create_bot_message("")
    history = history + [bot_response_msg]
    
    for character in response:
        history[-1]["content"] += character
        time.sleep(0.02)
        yield history, motion_uploaded, data_stored
    
    # Add video to chat only for text-to-motion tasks (not motion-to-text)
    if is_motion_generation:
        video_msg = create_video_message(output_mp4_path)
        history = history + [video_msg]
        yield history, motion_uploaded, data_stored


def bot_example(history, responses):
    """Append example responses to chatbot history (messages format)"""
    # Ensure both are lists
    if not isinstance(history, list):
        history = []
    if not isinstance(responses, list):
        responses = [responses]
    # Concatenate and return (messages format)
    return history + responses


with open("assets/css/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()

with gr.Blocks(css=customCSS) as demo:

    # Examples - converted to messages format
    chat_instruct = gr.State([
        create_bot_message("Hi, I'm MotionGPT! I can generate realistic human motion from text, or generate text from motion."),
        create_bot_message("You can chat with me in pure text like generating human motion following your descriptions."),
        create_bot_message("After generation, you can click the button in the top right of generation human motion result to download the human motion video or feature stored in .npy format."),
        create_bot_message("With the human motion feature file downloaded or got from dataset, you are able to ask me to translate it!"),
        create_bot_message("Of courser, you can also purely chat with me and let me give you human motion in text, here are some examples!"),
        create_bot_message("We provide two motion visulization methods. The default fast method is skeleton line ploting which is like the examples below:"),
        create_example_video("assets/videos/example0_fast.mp4"),
        create_bot_message("And the slow method is SMPL model rendering which is more realistic but slower."),
        create_example_video("assets/videos/example0.mp4"),
        create_bot_message("If you want to get the video in our paper and website like below, you can refer to the scirpt in our [github repo](https://github.com/OpenMotionLab/MotionGPT#-visualization)."),
        create_example_video("assets/videos/example0_blender.mp4"),
        create_bot_message("Follow the examples and try yourself!"),
    ])
    chat_instruct_sum = gr.State([create_bot_message('''Hi, I'm MotionGPT! I can generate realistic human motion from text, or generate text from motion.
         
         1. You can chat with me in pure text like generating human motion following your descriptions.
         2. After generation, you can click the button in the top right of generation human motion result to download the human motion video or feature stored in .npy format.
         3. With the human motion feature file downloaded or got from dataset, you are able to ask me to translate it!
         4. Of course, you can also purely chat with me and let me give you human motion in text, here are some examples!
         ''')] + chat_instruct.value[-7:])

    t2m_examples = gr.State([
        create_bot_message("You can chat with me in pure text, following are some examples of text-to-motion generation!"),
        create_user_message("A person is walking forwards, but stumbles and steps back, then carries on forward."),
        create_example_video("assets/videos/example0.mp4"),
        create_user_message("Generate a man aggressively kicks an object to the left using his right foot."),
        create_example_video("assets/videos/example1.mp4"),
        create_user_message("Generate a person lowers their arms, gets onto all fours, and crawls."),
        create_example_video("assets/videos/example2.mp4"),
        create_user_message("Show me the video of a person bends over and picks things up with both hands individually, then walks forward."),
        create_example_video("assets/videos/example3.mp4"),
        create_user_message("Imagine a person is practing balancing on one leg."),
        create_example_video("assets/videos/example5.mp4"),
        create_user_message("Show me a person walks forward, stops, turns directly to their right, then walks forward again."),
        create_example_video("assets/videos/example6.mp4"),
        create_user_message("I saw a person sits on the ledge of something then gets off and walks away."),
        create_example_video("assets/videos/example7.mp4"),
        create_user_message("Show me a person is crouched down and walking around sneakily."),
        create_example_video("assets/videos/example8.mp4"),
    ])

    m2t_examples = gr.State([
        create_bot_message("With the human motion feature file downloaded or got from dataset, you are able to ask me to translate it, here are some examples!"),
        create_user_message("Please explain the movement shown in <Motion_Placeholder> using natural language."),
        create_example_video("assets/videos/example0.mp4"),
        create_bot_message("The person was pushed but didn't fall down"),
        create_user_message("What kind of action is being represented in <Motion_Placeholder>? Explain it in text."),
        create_example_video("assets/videos/example4.mp4"),
        create_bot_message("The figure has its hands curled at jaw level, steps onto its left foot and raises right leg with bent knee to kick forward and return to starting stance."),
        create_user_message("Provide a summary of the motion demonstrated in <Motion_Placeholder> using words."),
        create_example_video("assets/videos/example2.mp4"),
        create_bot_message("A person who is standing with his arms up and away from his sides bends over, gets down on his hands and then his knees and crawls forward."),
        create_user_message("Generate text for <Motion_Placeholder>:"),
        create_example_video("assets/videos/example5.mp4"),
        create_bot_message("The man tries to stand in a yoga tree pose and looses his balance."),
        create_user_message("Provide a summary of the motion depicted in <Motion_Placeholder> using language."),
        create_example_video("assets/videos/example6.mp4"),
        create_bot_message("Person walks up some steps then leeps to the other side and goes up a few more steps and jumps dow"),
        create_user_message("Describe the motion represented by <Motion_Placeholder> in plain English."),
        create_example_video("assets/videos/example7.mp4"),
        create_bot_message("Person sits down, then stands up and walks forward. then the turns around 180 degrees and walks the opposite direction"),
        create_user_message("Provide a description of the action in <Motion_Placeholder> using words."),
        create_example_video("assets/videos/example8.mp4"),
        create_bot_message("This man is bent forward and walks slowly around."),
    ])

    t2t_examples = gr.State([
        create_bot_message("Of course, you can also purely chat with me and let me give you human motion in text, here are some examples!"),
        create_user_message('Depict a motion as like you have seen it.'),
        create_bot_message("A person slowly walked forward in rigth direction while making the circle"),
        create_user_message('Random say something about describing a human motion.'),
        create_bot_message("A man throws punches using his right hand."),
        create_user_message('Describe the motion of someone as you will.'),
        create_bot_message("Person is moving left to right in a dancing stance swaying hips, moving feet left to right with arms held out"),
        create_user_message('Come up with a human motion caption.'),
        create_bot_message("A person is walking in a counter counterclockwise motion."),
        create_user_message('Write a sentence about how someone might dance.'),
        create_bot_message("A person with his hands down by his sides reaches down for something with his right hand, uses the object to make a stirring motion, then places the item back down."),
        create_user_message('Depict a motion as like you have seen it.'),
        create_bot_message("A person is walking forward a few feet, then turns around, walks back, and continues walking.")
    ])

    # Convert messages to Gradio 4.0.0 format (list of tuples)
    def convert_to_tuples(messages):
        """Convert list of messages to list of tuples for Gradio 4.0.0"""
        result = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if isinstance(msg, tuple):
                # Already a tuple
                result.append(msg)
                i += 1
            elif isinstance(msg, dict):
                # Old format - skip (will be handled by new format)
                i += 1
            else:
                # String message - check if it's user or bot
                # For now, treat as bot message and pair with None user
                result.append((None, msg))
                i += 1
        return result
    
    # Combine examples and convert to tuple format for Gradio
    # Handle videos based on Gradio version (4.0.0 doesn't support dict format)
    Init_chatbot = (
        chat_instruct.value[:1]
        + t2m_examples.value[:3]
        + m2t_examples.value[:3]
        + t2t_examples.value[:2]
        + chat_instruct.value[-7:]
    )

    # Variables
    motion_uploaded = gr.State({
        "feats": None,
        "joints": None,
        "motion_video": None,
        "motion_lengths": 0,
        "motion_token": None,
        "motion_token_string": '',
        "motion_token_length": 0,
    })
    data_stored = gr.State([])

    gr.Markdown("# MotionGPT")

    chatbot = gr.Chatbot(Init_chatbot,
                         elem_id="mGPT",
                         height=600,
                         label="MotionGPT",
                         type="messages",
                         avatar_images=(None, "assets/images/avatar_bot.jpg"),
                         show_copy_button=True)

    with gr.Row():
        with gr.Column(scale=6):
            with gr.Row():
                txt = gr.Textbox(
                    label="Text",
                    show_label=False,
                    elem_id="textbox",
                    placeholder=
                    "Enter text and press ENTER or speak to input. You can also upload motion.",
                    container=False)

            with gr.Row():
                aud = gr.Audio(sources=["microphone"],
                               label="Speak input",
                               type='filepath')
                btn = gr.UploadButton("📁 Upload motion",
                                      elem_id="upload",
                                      file_types=["file"])
                # regen = gr.Button("🔄 Regenerate", elem_id="regen")
                clear = gr.ClearButton([txt, chatbot, aud], value='🗑️ Clear')

            with gr.Row():
                gr.Markdown('''
                ### You can get more examples (pre-generated for faster response) by clicking the buttons below:
                ''')

            with gr.Row():
                instruct_eg = gr.Button("Instructions", elem_id="instruct")
                t2m_eg = gr.Button("Text-to-Motion", elem_id="t2m")
                m2t_eg = gr.Button("Motion-to-Text", elem_id="m2t")
                t2t_eg = gr.Button("Random description", elem_id="t2t")

        with gr.Column(scale=1, min_width=150):
            method = gr.Dropdown(["slow", "fast"],
                                 label="Visualization method",
                                 interactive=True,
                                 elem_id="method",
                                 value="fast")

            language = gr.Dropdown(["English", "中文"],
                                   label="Speech language",
                                   interactive=True,
                                   elem_id="language",
                                   value="English")

    txt_msg = txt.submit(
        add_text, [chatbot, txt, motion_uploaded, data_stored, method],
        [chatbot, txt, motion_uploaded, data_stored],
        queue=False).then(bot, [chatbot, motion_uploaded, data_stored, method],
                          [chatbot, motion_uploaded, data_stored])

    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    file_msg = btn.upload(add_file, [chatbot, btn, txt, motion_uploaded],
                          [chatbot, txt, motion_uploaded],
                          queue=False)
    aud_msg = aud.stop_recording(
        add_audio, [chatbot, aud, data_stored, language],
        [chatbot, data_stored],
        queue=False).then(bot, [chatbot, motion_uploaded, data_stored, method],
                          [chatbot, motion_uploaded, data_stored])
    # regen_msg = regen.click(bot,
    #                         [chatbot, motion_uploaded, data_stored, method],
    #                         [chatbot, motion_uploaded, data_stored],
    #                         queue=False)

    instruct_msg = instruct_eg.click(bot_example, [chatbot, chat_instruct_sum],
                                     [chatbot],
                                     queue=False)
    t2m_eg_msg = t2m_eg.click(bot_example, [chatbot, t2m_examples], [chatbot],
                              queue=False)
    m2t_eg_msg = m2t_eg.click(bot_example, [chatbot, m2t_examples], [chatbot],
                              queue=False)
    t2t_eg_msg = t2t_eg.click(bot_example, [chatbot, t2t_examples], [chatbot],
                              queue=False)

    chatbot.change(scroll_to_output=True)

demo.queue()

# Disable API docs to avoid schema generation error (TypeError in gradio_client)
try:
    demo.api_open = False
except:
    pass

if __name__ == "__main__":
    # Detect HuggingFace Spaces environment
    is_hf_space = os.getenv("SPACE_ID") is not None
    
    if is_hf_space:
        # HuggingFace Spaces - use default settings
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    else:
        # Local deployment with ngrok
        try:
            from pyngrok import ngrok
            
            SERVER_PORT = 7860
            
            def start_ngrok():
                time.sleep(2)
                tunnel = ngrok.connect(SERVER_PORT)
                print(f"\n🌐 Public URL: {tunnel.public_url}")
                print("🔗 Share this URL to access your MotionGPT app from anywhere!")
            
            ngrok_thread = threading.Thread(target=start_ngrok)
            ngrok_thread.daemon = True
            ngrok_thread.start()
            
            demo.launch(server_name="0.0.0.0", server_port=SERVER_PORT, share=True, debug=True)
        except ImportError:
            # Fallback to Gradio share if ngrok not available
            demo.launch(server_name="0.0.0.0", server_port=7860, share=True, debug=True)
