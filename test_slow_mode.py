#!/usr/bin/env python3
"""
Test script for slow mode (SMPL rendering) without Gradio
This helps diagnose why slow mode isn't working
"""

import os
import sys
import numpy as np
import torch

# Fix chumpy compatibility with NumPy 1.23+ (MUST be before chumpy import)
import numpy
numpy.bool = numpy.bool_
numpy.int = numpy.int_
numpy.float = numpy.float_
numpy.complex = numpy.complex_
numpy.object = numpy.object_
numpy.unicode = numpy.str_
numpy.str = numpy.str_
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.complex = np.complex_
np.object = np.object_
np.unicode = np.str_
np.str = np.str_

print("=" * 60)
print("Testing Slow Mode (SMPL Rendering) Dependencies")
print("=" * 60)

# Test 1: Check chumpy
print("\n1. Testing chumpy import...")
try:
    import chumpy
    print("   ✅ chumpy imported successfully")
except Exception as e:
    print(f"   ❌ chumpy import failed: {e}")
    print("   Try: pip install --no-build-isolation chumpy")
    sys.exit(1)

# Test 2: Check SMPLRender
print("\n2. Testing SMPLRender import...")
try:
    from mGPT.render.pyrender.smpl_render import SMPLRender
    print("   ✅ SMPLRender imported successfully")
except Exception as e:
    print(f"   ❌ SMPLRender import failed: {e}")
    sys.exit(1)

# Test 3: Check config and SMPL model path
print("\n3. Testing config and SMPL model path...")
try:
    from mGPT.config import parse_args
    cfg = parse_args(phase="webui")
    smpl_model_path = cfg.RENDER.SMPL_MODEL_PATH
    print(f"   Config loaded: ✅")
    print(f"   SMPL_MODEL_PATH: {smpl_model_path}")
    print(f"   Absolute path: {os.path.abspath(smpl_model_path)}")
    
    # Check if path exists
    if os.path.exists(smpl_model_path):
        print(f"   ✅ Path exists")
        
        # Check for required files (SMPL expects parent directory with SMPL_NEUTRAL.pkl)
        parent_dir = os.path.dirname(smpl_model_path) if os.path.isfile(smpl_model_path) else smpl_model_path
        parent_dir = os.path.dirname(parent_dir) if os.path.basename(parent_dir) == 'smpl' else parent_dir
        
        print(f"   Checking parent directory: {parent_dir}")
        if os.path.exists(parent_dir):
            print(f"   ✅ Parent directory exists")
            # List files in parent
            if os.path.isdir(parent_dir):
                files = os.listdir(parent_dir)
                print(f"   Files in parent: {files[:10]}...")  # Show first 10
                
                # Check for SMPL model files
                smpl_files = [f for f in files if 'SMPL' in f.upper() or f.endswith('.pkl')]
                if smpl_files:
                    print(f"   ✅ Found SMPL model files: {smpl_files[:5]}")
                else:
                    print(f"   ⚠️  No obvious SMPL model files found")
        else:
            print(f"   ❌ Parent directory does not exist: {parent_dir}")
    else:
        print(f"   ❌ Path does not exist!")
        print(f"   Expected: {os.path.abspath(smpl_model_path)}")
        print(f"   Current working directory: {os.getcwd()}")
        
except Exception as e:
    print(f"   ❌ Config loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check OpenGL/EGL setup
print("\n4. Testing OpenGL/EGL setup...")
os.environ['DISPLAY'] = ':0.0'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
print(f"   DISPLAY: {os.environ.get('DISPLAY')}")
print(f"   PYOPENGL_PLATFORM: {os.environ.get('PYOPENGL_PLATFORM')}")

try:
    import pyrender
    print("   ✅ pyrender imported successfully")
except Exception as e:
    print(f"   ❌ pyrender import failed: {e}")
    print("   Try: pip install pyrender")

# Test 5: Try creating SMPLRender instance
print("\n5. Testing SMPLRender instantiation...")
render = None
try:
    # Force CPU to avoid CUDA compatibility issues (PyTorch may not support old GPUs)
    original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA to force CPU
    try:
        render = SMPLRender(smpl_model_path)
        print("   ✅ SMPLRender created successfully")
        print(f"   Device: {render.device}")
        # Ensure it's using CPU
        if render.device.type == 'cuda':
            print("   ⚠️  Warning: Renderer is using CUDA, but may fail on older GPUs")
            print("   Forcing CPU...")
            render.device = torch.device("cpu")
            render.smpl = render.smpl.cpu()
            print(f"   ✅ Forced to CPU: {render.device}")
    finally:
        # Restore original setting
        if original_cuda_visible is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
except Exception as e:
    print(f"   ❌ SMPLRender creation failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n   Common issues:")
    print("   - SMPL model files not found at expected path")
    print("   - SMPL model path should point to directory containing SMPL_NEUTRAL.pkl")
    print("   - Check that deps/smpl_models/smpl/ exists with SMPL_NEUTRAL.pkl")
    sys.exit(1)

# Test 6: Test rendering with dummy data
print("\n6. Testing rendering with dummy data...")
try:
    from scipy.spatial.transform import Rotation as RRR
    from mGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
    
    # Create dummy joint data (typical shape: [frames, joints, 3])
    print("   Creating dummy joint data...")
    dummy_data = np.random.randn(10, 22, 3).astype(np.float32)  # 10 frames, 22 joints, 3D
    dummy_data = dummy_data - dummy_data[0, 0]  # Center at origin
    
    print("   Converting joints to poses...")
    pose_generator = HybrIKJointsToRotmat()
    pose = pose_generator(dummy_data)
    pose = np.concatenate([
        pose,
        np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)
    ], 1)
    
    print("   Initializing renderer...")
    shape = [768, 768]
    r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
    pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])
    aroot = dummy_data[[0], 0]
    aroot[:, 1] = -aroot[:, 1]
    params = dict(pred_shape=np.zeros([1, 10]),
                  pred_root=aroot,
                  pred_pose=pose)
    
    render.init_renderer([shape[0], shape[1], 3], params)
    print("   ✅ Renderer initialized")
    
    print("   Rendering first frame...")
    renderImg = render.render(0)
    print(f"   ✅ Frame rendered successfully! Shape: {renderImg.shape}")
    print(f"   Image dtype: {renderImg.dtype}, min: {renderImg.min():.3f}, max: {renderImg.max():.3f}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED! Slow mode should work.")
    print("=" * 60)
    
except Exception as e:
    print(f"   ❌ Rendering test failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n   Common issues:")
    print("   - OpenGL/EGL not properly configured (headless server?)")
    print("   - GPU/display issues")
    print("   - SMPL model loading errors")
    sys.exit(1)

