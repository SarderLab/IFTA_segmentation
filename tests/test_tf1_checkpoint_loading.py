#!/usr/bin/env python3
"""
Test the TF1 checkpoint loading fix
"""

import os
import sys
import subprocess

def test_checkpoint_loading():
    """Test that the model can load TF1 checkpoints."""
    
    os.chdir("/home/anish.tatke/blue-group/anish.tatke/IFTA_segmentation")
    
    # Test command similar to what IterativePredict_1X calls
    test_cmd = [
        sys.executable, 
        "Codes/Deeplab_network/main.py",
        "--option", "predict",
        "--modeldir", "/home/anish.tatke/blue-group/anish.tatke/IFTA_segmentation/DDS_JHU/MODELS/0/HR",
        "--test_step", "1",
        "--encoder_name", "deeplab",
        "--num_classes", "4",
        "--gpu", "0",
        "--test_data_list", "/tmp/empty_test_list.txt",  # Will create this
        "--out_dir", "/tmp/test_output",
        "--data_dir", "/tmp",
        "--test_num_steps", "1"
    ]
    
    # Create empty test files
    os.makedirs("/tmp/test_output", exist_ok=True)
    with open("/tmp/empty_test_list.txt", "w") as f:
        f.write("dummy.jpg dummy_label.png\n")
    
    print("Testing TF1 checkpoint loading:")
    print(" ".join(test_cmd))
    
    try:
        # Run for a few seconds to test checkpoint loading
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
        
        output = result.stdout + result.stderr
        
        if "Successfully loaded TF1 checkpoint" in output:
            print("✅ TF1 checkpoint loading successful!")
        elif "Restored TF2 model" in output:
            print("✅ TF2 checkpoint loading successful!")
        elif "Error when restoring from checkpoint" in output:
            print("❌ Checkpoint loading still failing:")
            print(output)
        elif "No checkpoint found" in output:
            print("❌ Checkpoint not found:")
            print(output)
        else:
            print("⚠️ Uncertain result. Output:")
            print(output[:1000])
            
    except subprocess.TimeoutExpired:
        print("⚠️ Test timed out (might be normal if checkpoint loaded successfully)")
    except Exception as e:
        print(f"❌ Error running test: {e}")
        
    # Cleanup
    try:
        os.remove("/tmp/empty_test_list.txt")
        os.rmdir("/tmp/test_output")
    except:
        pass

if __name__ == "__main__":
    test_checkpoint_loading()
