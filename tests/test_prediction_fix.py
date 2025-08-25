#!/usr/bin/env python3
"""
Test the actual prediction pipeline to make sure the import fix works
"""

import subprocess
import sys
import os

def test_prediction_pipeline():
    """Test a minimal prediction to verify the import fix."""
    
    # Set up environment
    os.chdir("/home/anish.tatke/blue-group/anish.tatke/IFTA_segmentation")
    
    # Test the exact command that would be called by IterativePredict_1X.py
    main_py_path = "/home/anish.tatke/blue-group/anish.tatke/IFTA_segmentation/Codes/Deeplab_network/main.py"
    
    test_cmd = [
        sys.executable, main_py_path,
        '--option', 'predict',
        '--encoder_name', 'deeplab',
        '--gpu', '0',
        '--num_classes', '4'
    ]
    
    print("Testing prediction command:")
    print(" ".join(test_cmd))
    
    try:
        # Run for a few seconds to see if imports work
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=5)
        
        if "ImportError" in result.stderr:
            print("❌ Import error still present:")
            print(result.stderr)
        elif "attempted relative import" in result.stderr:
            print("❌ Relative import error still present:")
            print(result.stderr)
        else:
            print("✅ No import errors detected!")
            if result.stderr:
                print("Other output/warnings:")
                print(result.stderr[:500])
                
    except subprocess.TimeoutExpired:
        print("✅ Command timed out (expected) - no import errors!")
    except Exception as e:
        print(f"❌ Error running command: {e}")

if __name__ == "__main__":
    test_prediction_pipeline()
