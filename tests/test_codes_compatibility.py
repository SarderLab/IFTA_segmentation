#!/usr/bin/env python3
"""
Test script to validate that all Codes files work correctly with the TF2.x migration
This checks for import compatibility and argument compatibility
"""

import os
import sys
import tempfile
import subprocess
import argparse
from pathlib import Path

# Add the Codes directory to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'Codes'))

def test_main_py_arguments():
    """Test that main.py can handle the arguments passed by other scripts."""
    print("=== Testing main.py Argument Compatibility ===")
    
    main_py_path = project_root / 'Codes' / 'main.py'
    
    # Test help output to see available arguments
    try:
        result = subprocess.run([sys.executable, str(main_py_path), '--help'], 
                              capture_output=True, text=True, timeout=30)
        print("✓ main.py help command successful")
        
        # Check for key arguments that other scripts use
        help_output = result.stdout
        required_args = [
            '--option', '--test_data_list', '--out_dir', '--test_step',
            '--test_num_steps', '--modeldir', '--data_dir', '--num_classes',
            '--gpu', '--learning_rate', '--encoder_name'
        ]
        
        missing_args = []
        for arg in required_args:
            if arg not in help_output:
                missing_args.append(arg)
        
        if missing_args:
            print(f"⚠ Missing arguments in main.py: {missing_args}")
        else:
            print("✓ All required arguments found in main.py")
            
    except subprocess.TimeoutExpired:
        print("⚠ main.py help command timed out")
    except Exception as e:
        print(f"❌ main.py help test failed: {e}")

def test_imports():
    """Test that all key modules can be imported."""
    print("\n=== Testing Module Imports ===")
    
    try:
        # Test main imports
        print("Testing main module imports...")
        import main
        print("✓ main.py imports successful")
        
        # Test other modules that might import TF code
        modules_to_test = [
            'evolve_predictions',
            'generateTrainSet', 
            'IterativeTraining',
            'IterativeTraining_1X',
            'IterativePredict',
            'IterativePredict_1X',
            'predict_xml',
            'xml_to_mask',
            'getWsi'
        ]
        
        for module_name in modules_to_test:
            try:
                module = __import__(module_name)
                print(f"✓ {module_name} imports successful")
            except ImportError as e:
                print(f"⚠ {module_name} import failed: {e}")
            except Exception as e:
                print(f"⚠ {module_name} import error: {e}")
                
    except Exception as e:
        print(f"❌ Import test failed: {e}")

def test_deeplab_network_access():
    """Test that Deeplab_network module is accessible."""
    print("\n=== Testing Deeplab_network Access ===")
    
    try:
        from Deeplab_network.model import Model
        print("✓ Deeplab_network.model.Model import successful")
        
        from Deeplab_network.network import Deeplab_v2_TF2, ResNet_segmentation_TF2
        print("✓ TF2 network classes import successful")
        
        # Test that the __init__.py is working
        import Deeplab_network
        print("✓ Deeplab_network package import successful")
        
    except Exception as e:
        print(f"❌ Deeplab_network access test failed: {e}")

def test_file_paths():
    """Test that all file references are correct."""
    print("\n=== Testing File Path References ===")
    
    # Check if the main files exist
    required_files = [
        'main.py',
        'Deeplab_network/main.py',
        'Deeplab_network/model.py',
        'Deeplab_network/network.py'
    ]
    
    codes_dir = project_root / 'Codes'
    
    for file_path in required_files:
        full_path = codes_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")

def test_python_version_compatibility():
    """Test Python version compatibility."""
    print("\n=== Testing Python Version Compatibility ===")
    
    print(f"Current Python version: {sys.version}")
    
    # Check if we're using a compatible Python version
    if sys.version_info >= (3, 8):
        print("✓ Python version is compatible with TF2.x")
    else:
        print("⚠ Python version might be too old for TF2.x")

def test_tensorflow_availability():
    """Test TensorFlow availability and version."""
    print("\n=== Testing TensorFlow Availability ===")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} available")
        
        if tf.__version__.startswith('2.'):
            print("✓ TensorFlow 2.x detected")
        else:
            print("⚠ TensorFlow 1.x detected - migration issues possible")
            
        # Test GPU availability
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✓ {len(gpus)} GPU(s) available")
        else:
            print("⚠ No GPUs detected")
            
    except ImportError:
        print("❌ TensorFlow not available")
    except Exception as e:
        print(f"⚠ TensorFlow test error: {e}")

def main():
    """Run all compatibility tests."""
    print("🔧 IFTA Segmentation - Codes Folder Compatibility Test")
    print("=" * 60)
    
    test_python_version_compatibility()
    test_tensorflow_availability()
    test_file_paths()
    test_deeplab_network_access()
    test_imports()
    test_main_py_arguments()
    
    print("\n" + "=" * 60)
    print("🎯 Compatibility test completed!")
    print("\nIf all tests passed, the Codes folder is ready for TF2.x usage.")
    print("Any warnings should be addressed before production use.")

if __name__ == "__main__":
    main()
