#!/usr/bin/env python3
"""
Test script to validate the TF2 network implementations
"""

import os
import sys
import tensorflow as tf
import numpy as np

# Add the Codes directory to the path
sys.path.insert(0, '/home/anish.tatke/blue-group/anish.tatke/IFTA_segmentation/Codes')

# Configure TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_tf2_networks():
    """Test both TF2 network implementations"""
    print("Testing TF2 Network Implementations...")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Test parameters
    num_classes = 21  # VOC has 21 classes including background
    batch_size = 1
    input_height = 256
    input_width = 256
    channels = 3
    
    # Create dummy input
    dummy_input = tf.random.normal((batch_size, input_height, input_width, channels))
    print(f"Input shape: {dummy_input.shape}")
    
    try:
        # Test Deeplab_v2_TF2
        print("\n=== Testing Deeplab_v2_TF2 ===")
        from Deeplab_network.network import Deeplab_v2_TF2
        
        deeplab_model = Deeplab_v2_TF2(num_classes=num_classes)
        print("✓ Deeplab_v2_TF2 instantiated successfully")
        
        # Test forward pass
        with tf.device('/CPU:0'):  # Use CPU for consistency
            deeplab_output = deeplab_model(dummy_input, training=False)
        
        print(f"✓ Deeplab forward pass successful!")
        print(f"  - Output shape: {deeplab_output.shape}")
        print(f"  - Output dtype: {deeplab_output.dtype}")
        print(f"  - Output min/max: {tf.reduce_min(deeplab_output):.3f} / {tf.reduce_max(deeplab_output):.3f}")
        
    except Exception as e:
        print(f"✗ Deeplab_v2_TF2 test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Test ResNet_segmentation_TF2
        print("\n=== Testing ResNet_segmentation_TF2 ===")
        from Deeplab_network.network import ResNet_segmentation_TF2
        
        resnet_model = ResNet_segmentation_TF2(num_classes=num_classes)
        print("✓ ResNet_segmentation_TF2 instantiated successfully")
        
        # Test forward pass
        with tf.device('/CPU:0'):  # Use CPU for consistency
            resnet_output = resnet_model(dummy_input, training=False)
        
        print(f"✓ ResNet forward pass successful!")
        print(f"  - Output shape: {resnet_output.shape}")
        print(f"  - Output dtype: {resnet_output.dtype}")
        print(f"  - Output min/max: {tf.reduce_min(resnet_output):.3f} / {tf.reduce_max(resnet_output):.3f}")
        
    except Exception as e:
        print(f"✗ ResNet_segmentation_TF2 test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test model summary
    try:
        print("\n=== Model Summaries ===")
        print("Deeplab_v2_TF2 trainable parameters:", deeplab_model.count_params())
        print("ResNet_segmentation_TF2 trainable parameters:", resnet_model.count_params())
    except Exception as e:
        print(f"Parameter counting failed: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_tf2_networks()
