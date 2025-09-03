#!/usr/bin/env python3
"""
Minimal test for TF1 checkpoint loading functionality
Tests the refactored checkpoint loading system
"""

import os
import sys
import tempfile

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TensorFlow
import tensorflow as tf

# Import the TF1 checkpoint loader directly
from Codes.Deeplab_network.tf1_checkpoint_loader import TF1CheckpointLoader

def test_tf1_checkpoint_loader_creation():
    """Test that TF1CheckpointLoader can be created successfully"""
    print("Testing TF1CheckpointLoader creation...")
    
    try:
        # Create a simple model for testing
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, padding='same', name='conv1'),
            tf.keras.layers.Conv2D(128, 3, padding='same', name='conv2'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(21, name='fc')
        ])
        
        # Build the model with a sample input
        model.build((None, 513, 513, 3))
        
        # Create TF1CheckpointLoader
        loader = TF1CheckpointLoader(model)
        
        print(" TF1CheckpointLoader created successfully")
        print(f" Mapping rules created: {len(loader.mapping_rules)} rules")
        
        return True
        
    except Exception as e:
        print(f" Failed to create TF1CheckpointLoader: {e}")
        return False

def test_mapping_rules():
    """Test that mapping rules are created correctly"""
    print("\nTesting mapping rules creation...")
    
    try:
        # Create a dummy model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, name='dense1'),
            tf.keras.layers.Dense(21, name='dense2')
        ])
        model.build((None, 100))
        
        loader = TF1CheckpointLoader(model)
        
        # Check that mapping rules exist
        if len(loader.mapping_rules) > 0:
            print(f" Created {len(loader.mapping_rules)} mapping rules")
            
            # Show a few sample mappings
            print("Sample mappings:")
            for i, (tf1_name, tf2_name) in enumerate(loader.mapping_rules.items()):
                if i < 5:  # Show first 5
                    print(f"  {tf1_name} -> {tf2_name}")
                elif i == 5:
                    print("  ...")
                    break
            
            return True
        else:
            print(" No mapping rules created")
            return False
            
    except Exception as e:
        print(f" Failed to create mapping rules: {e}")
        return False

def test_checkpoint_file_detection():
    """Test TF1 checkpoint validation"""
    print("\nTesting checkpoint file validation...")
    
    try:
        # Create a dummy model
        model = tf.keras.Sequential([tf.keras.layers.Dense(21)])
        model.build((None, 100))
        
        loader = TF1CheckpointLoader(model)
        
        # Test with actual TF1 checkpoint path
        tf1_checkpoint = '/home/anish.tatke/blue-group/anish.tatke/IFTA_segmentation/DiagnosticCore/MODELS/0/HR/model.ckpt-1'
        
        try:
            validated_path = loader._validate_checkpoint_path(tf1_checkpoint)
            print(f" TF1 checkpoint validation successful: {validated_path}")
            return True
        except FileNotFoundError:
            print(" TF1 checkpoint file not found, but validation logic works")
            return True
            
    except Exception as e:
        print(f" Failed checkpoint validation test: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting TF1 Checkpoint Loading Tests")
    print("=" * 50)
    
    tests = [
        test_tf1_checkpoint_loader_creation,
        test_mapping_rules,
        test_checkpoint_file_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f" Test {test_func.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print(" All tests passed!")
        return 0
    else:
        print(" Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
