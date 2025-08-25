#!/usr/bin/env python3
"""
Test script to verify that the Deeplab_network main.py can be called correctly
"""

import subprocess
import sys
import os

def test_main_py_import():
    """Test that main.py can be imported/called without import errors."""
    print("Testing Deeplab_network/main.py import fix...")
    
    # Test 1: Try to run main.py with --help to see if imports work
    main_py_path = "/home/anish.tatke/blue-group/anish.tatke/IFTA_segmentation/Codes/Deeplab_network/main.py"
    
    try:
        # Change to the project root directory
        os.chdir("/home/anish.tatke/blue-group/anish.tatke/IFTA_segmentation")
        
        result = subprocess.run([sys.executable, main_py_path, "--help"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ main.py imports successfully!")
            print("Available arguments:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        else:
            print("❌ main.py failed with error:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⚠️ main.py help command timed out")
    except Exception as e:
        print(f"❌ Error testing main.py: {e}")

    # Test 2: Try to import the Model class directly
    try:
        sys.path.insert(0, "/home/anish.tatke/blue-group/anish.tatke/IFTA_segmentation/Codes/Deeplab_network")
        from model import Model
        print("✅ Model class imports successfully!")
    except Exception as e:
        print(f"❌ Model import failed: {e}")

if __name__ == "__main__":
    test_main_py_import()
