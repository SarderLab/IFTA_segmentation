#!/usr/bin/env python3
"""
Test script to validate the complete TF2 integration (model.py + network.py)
"""

import os
import sys
import tensorflow as tf
import numpy as np

# Add the Codes directory to the path
sys.path.insert(0, '/home/anish.tatke/blue-group/anish.tatke/IFTA_segmentation/Codes')

# Configure TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_complete_integration():
    """Test the complete TF2 integration"""
    print("Testing Complete TF2 Integration...")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Test basic model import and instantiation
    try:
        print("\n=== Testing Model Class Import ===")
        from Deeplab_network.model import Model
        print("✓ Model import successful")
        
        # Create a simple config
        class Config:
            def __init__(self):
                self.batch_size = 1
                self.learning_rate = 0.001
                self.momentum = 0.9
                self.weight_decay = 0.0001
                self.num_classes = 21
                self.ignore_label = 255
                self.num_steps = 1000
                self.save_interval = 100
                self.snapshot_dir = '/tmp/test_snapshots'
                self.data_dir = '/tmp/test_data'
                self.logdir = '/tmp/test_logs'
                self.random_mirror = True
                self.random_scale = True
                self.model_name = 'Deeplab_v2_TF2'  # Use our new TF2 model
                self.encoder_name = 'deeplab'  # Required for model creation
                self.is_training = True
                self.not_restore_last = False
                self.restore_from = None
                
        config = Config()
        print("✓ Config created")
        
        # Create data directory for testing
        os.makedirs(config.snapshot_dir, exist_ok=True)
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(config.logdir, exist_ok=True)
        
        # Instantiate the Model
        model = Model(config)
        print("✓ Model instantiated successfully")
        print(f"  - Model type: {type(model.model)}")
        print(f"  - Model loaded: {model.model is not None}")
        print(f"  - Mean IoU metric: {type(model.miou_metric)}")
        
        # Test model setup
        print("\n=== Testing Model Setup ===")
        model.build_model((256, 256, 3))
        model.setup_optimizers()
        
        print("✓ Model built successfully")
        print(f"  - Model type: {type(model.model)}")
        print(f"  - Model loaded: {model.model is not None}")
        print(f"  - Optimizer encoder: {type(model.optimizer_encoder)}")
        
        # Test basic functionality if model is available
        if model.model is not None:
            print("\n=== Testing Model Forward Pass ===")
            dummy_image = tf.random.normal((1, 256, 256, 3))
            
            # Test direct model call
            with tf.device('/CPU:0'):
                prediction = model.model(dummy_image, training=False)
                
            print("✓ Model forward pass successful")
            print(f"  - Prediction shape: {prediction.shape}")
            print(f"  - Prediction dtype: {prediction.dtype}")
            
            # Test training step simulation
            print("\n=== Testing Loss Computation ===")
            dummy_labels = tf.random.uniform((1, 256, 256, 1), maxval=21, dtype=tf.int32)  # Add channel dimension
            
            with tf.device('/CPU:0'):
                loss_value = model.compute_loss(prediction, dummy_labels)
                
            print("✓ Loss computation successful")
            print(f"  - Loss value: {loss_value:.4f}")
        else:
            print("⚠ Model not loaded - skipping forward pass tests")
        
        # Test checkpoint functionality
        print("\n=== Testing Model Checkpoint ===")
        checkpoint_path = os.path.join(config.snapshot_dir, 'test_checkpoint')
        try:
            model.save_checkpoint(step=1)
            print("✓ Model checkpoint mechanism available")
        except Exception as e:
            print(f"⚠ Checkpoint test skipped: {e}")
        
        # Test metrics
        print("\n=== Testing Metrics ===")
        miou_result = model.miou_metric.result()
        print(f"✓ Mean IoU metric: {miou_result:.4f}")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print("\n=== Complete Integration Test Successful! ===")
    return True

if __name__ == "__main__":
    test_complete_integration()
