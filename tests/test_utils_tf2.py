#!/usr/bin/env python3
"""
Test script to validate all TF2.x utility functions
"""

import os
import sys
import tempfile
import tensorflow as tf
import numpy as np
from PIL import Image

# Add the Codes directory to the path
sys.path.insert(0, '/home/anish.tatke/blue-group/anish.tatke/IFTA_segmentation/Codes')

# Configure TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_dummy_data(temp_dir):
    """Create dummy image and label data for testing."""
    # Create dummy images and labels
    img_dir = os.path.join(temp_dir, 'images')
    label_dir = os.path.join(temp_dir, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    # Create dummy RGB image
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img_path = os.path.join(img_dir, 'test_image.jpg')
    Image.fromarray(dummy_img).save(img_path)
    
    # Create dummy segmentation mask
    dummy_label = np.random.randint(0, 21, (256, 256), dtype=np.uint8)
    label_path = os.path.join(label_dir, 'test_label.png')
    Image.fromarray(dummy_label).save(label_path)
    
    # Create data list file
    data_list_path = os.path.join(temp_dir, 'data_list.txt')
    with open(data_list_path, 'w') as f:
        f.write(f"/images/test_image.jpg /labels/test_label.png\n")
    
    return temp_dir, data_list_path

def test_utility_functions():
    """Test all utility functions."""
    print("Testing TF2.x Utility Functions...")
    print(f"TensorFlow version: {tf.__version__}")
    
    try:
        # Test imports
        print("\n=== Testing Imports ===")
        from Deeplab_network.utils import (
            ImageReader, read_labeled_image_list, create_tf2_dataset,
            decode_labels, inv_preprocess, prepare_label,
            write_log, write_tf_summary, write_image_summary, write_histogram_summary
        )
        print("✓ All utility imports successful")
        
        # Create temporary test data
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir, data_list_path = create_dummy_data(temp_dir)
            
            # Test read_labeled_image_list
            print("\n=== Testing read_labeled_image_list ===")
            image_list, label_list = read_labeled_image_list(data_dir, data_list_path)
            print(f"✓ Read {len(image_list)} images and {len(label_list)} labels")
            
            # Test ImageReader (legacy interface)
            print("\n=== Testing ImageReader (TF2 compatible) ===")
            img_mean = tf.constant([104.00698793, 116.66876762, 122.67891434])
            reader = ImageReader(
                data_dir=data_dir,
                data_list=data_list_path,
                input_size=(256, 256),
                random_scale=True,
                random_mirror=True,
                ignore_label=255,
                img_mean=img_mean,
                coord=None  # TF2 doesn't need coordinator
            )
            print("✓ ImageReader instantiated successfully")
            
            # Test dataset creation
            dataset = reader.get_dataset()
            print(f"✓ Dataset created: {dataset}")
            
            # Test batch creation
            image_batch, label_batch = reader.dequeue(2)
            print(f"✓ Batch created - Images: {image_batch.shape}, Labels: {label_batch.shape}")
            
            # Test create_tf2_dataset (new TF2 interface)
            print("\n=== Testing create_tf2_dataset ===")
            tf2_dataset = create_tf2_dataset(
                data_dir=data_dir,
                data_list=data_list_path,
                input_size=(256, 256),
                batch_size=2,
                random_scale=True,
                random_mirror=True,
                ignore_label=255,
                img_mean=img_mean,
                shuffle=True,
                repeat=False
            )
            print(f"✓ TF2 dataset created: {tf2_dataset}")
            
            # Test iterating over dataset
            for batch_images, batch_labels in tf2_dataset.take(1):
                print(f"✓ Dataset iteration - Images: {batch_images.shape}, Labels: {batch_labels.shape}")
                test_images = batch_images
                test_labels = batch_labels
                break
            
            # Test label utilities
            print("\n=== Testing Label Utilities ===")
            
            # Test prepare_label
            prepared_labels = prepare_label(
                test_labels, 
                new_size=[128, 128], 
                num_classes=21, 
                one_hot=False
            )
            print(f"✓ prepare_label - Output shape: {prepared_labels.shape}")
            
            # Test decode_labels
            predicted_labels = tf.argmax(tf.random.normal((2, 128, 128, 21)), axis=-1)
            predicted_labels = tf.expand_dims(predicted_labels, -1)
            decoded = decode_labels(predicted_labels.numpy(), num_images=1, num_classes=21)
            print(f"✓ decode_labels - Output shape: {decoded.shape}")
            
            # Test inv_preprocess
            processed_imgs = inv_preprocess(
                test_images.numpy(), 
                num_images=1, 
                img_mean=img_mean.numpy()
            )
            print(f"✓ inv_preprocess - Output shape: {processed_imgs.shape}")
            
            # Test logging utilities
            print("\n=== Testing Logging Utilities ===")
            
            log_file = os.path.join(temp_dir, 'test.log')
            write_log("Test log message", log_file)
            print("✓ write_log successful")
            
            # Test TF2 summary writing
            log_dir = os.path.join(temp_dir, 'logs')
            summary_writer = tf.summary.create_file_writer(log_dir)
            
            write_tf_summary(summary_writer, 'test_loss', 0.5, step=1)
            print("✓ write_tf_summary successful")
            
            write_image_summary(summary_writer, 'test_images', test_images[:1], step=1)
            print("✓ write_image_summary successful")
            
            test_values = tf.random.normal((100,))
            write_histogram_summary(summary_writer, 'test_histogram', test_values, step=1)
            print("✓ write_histogram_summary successful")
            
            summary_writer.close()
            
    except Exception as e:
        print(f"✗ Utility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print("\n=== All Utility Functions Test Successful! ===")
    return True

if __name__ == "__main__":
    test_utility_functions()
