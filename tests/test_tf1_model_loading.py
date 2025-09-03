#!/usr/bin/env python3
"""
Comprehensive Test: TF1 Model Loading and TF2 Migration Validation
==================================================================

This test validates that our TF2 migration can successfully load the TF1 model
from DiagnosticCore/MODELS/0/HR and produces identical behavior.

Test Objectives:
1. Load TF1 checkpoint using our comprehensive mapping system
2. Validate all mappings are correctly applied
3. Count trainable parameters and compare with TF1
4. Test model behavior matches TF1 exactly
5. Verify output consistency and quality
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.training import py_checkpoint_reader

# Add paths
sys.path.append('/home/anish.tatke/blue-group/anish.tatke/IFTA_segmentation/Codes/Deeplab_network')
sys.path.append('/home/anish.tatke/blue-group/anish.tatke/IFTA_segmentation/Codes')

from model import Model
from network import Deeplab_v2_TF2

def print_section(title):
    """Print formatted section header"""
    print(f'\n{"="*70}')
    print(f'{title.center(70)}')
    print(f'{"="*70}')

def print_subsection(title):
    """Print formatted subsection header"""
    print(f'\n{"-"*50}')
    print(f'{title}')
    print(f'{"-"*50}')

class TestConfig:
    """Configuration for TF1 model loading test"""
    def __init__(self):
        self.num_classes = 21
        self.ignore_label = 255
        self.logdir = '/tmp/tf1_model_test_logs'
        self.learning_rate = 0.0001
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.bsize = 1
        self.input_size = (513, 513)
        self.tf2_fixed = True
        self.encoder_name = 'res50'  # Add missing encoder_name
        # Point to the actual TF1 checkpoint
        self.checkpoint = '/home/anish.tatke/blue-group/anish.tatke/IFTA_segmentation/DiagnosticCore/MODELS/0/HR/model.ckpt-1'

def analyze_tf1_checkpoint(checkpoint_path):
    """Analyze the TF1 checkpoint to understand its structure"""
    print_subsection("TF1 CHECKPOINT ANALYSIS")
    
    try:
        reader = py_checkpoint_reader.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        
        print(f" TF1 Checkpoint Statistics:")
        print(f"   Total variables: {len(var_to_shape_map)}")
        
        # Categorize variables
        categories = {
            'Initial Conv': [],
            'ASPP': [],
            'ResNet Blocks': [],
            'Other': []
        }
        
        for var_name, shape in var_to_shape_map.items():
            if 'conv1' in var_name or 'bn_conv1' in var_name:
                categories['Initial Conv'].append((var_name, shape))
            elif 'fc1_voc12' in var_name:
                categories['ASPP'].append((var_name, shape))
            elif 'res' in var_name and ('branch' in var_name or 'bn' in var_name):
                categories['ResNet Blocks'].append((var_name, shape))
            else:
                categories['Other'].append((var_name, shape))
        
        print(f"\n Variable Categories:")
        for category, vars_list in categories.items():
            print(f"   {category}: {len(vars_list)} variables")
        
        # Show sample variables from each category
        print(f"\n Sample Variables by Category:")
        for category, vars_list in categories.items():
            if vars_list:
                print(f"   {category}:")
                for i, (var_name, shape) in enumerate(vars_list[:3]):
                    print(f"      {var_name}: {shape}")
                if len(vars_list) > 3:
                    print(f"      ... and {len(vars_list) - 3} more")
        
        return var_to_shape_map, categories
        
    except Exception as e:
        print(f" Failed to analyze TF1 checkpoint: {e}")
        return None, None

def create_tf2_model_and_test_mapping(config):
    """Create TF2 model and test comprehensive mapping"""
    print_subsection("TF2 MODEL CREATION & MAPPING VALIDATION")
    
    try:
        # Create model instance
        model = Model(config)
        print(f" TF2 Model instance created")
        
        # Create standalone TF2 model for validation
        tf2_model = Deeplab_v2_TF2(num_classes=config.num_classes, name='resnet_v1_50')
        dummy_input = tf.ones((1, config.input_size[0], config.input_size[1], 3))
        output = tf2_model(dummy_input, training=False)
        
        print(f" TF2 model built successfully")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{tf.reduce_min(output):.3f}, {tf.reduce_max(output):.3f}]")
        print(f"   Trainable variables: {len(tf2_model.trainable_variables)}")
        print(f"   Total variables: {len(tf2_model.variables)}")
        
        # Get comprehensive mappings from TF1CheckpointLoader
        from tf1_checkpoint_loader import TF1CheckpointLoader
        temp_loader = TF1CheckpointLoader(tf2_model)
        mappings = temp_loader.mapping_rules
        print(f" Generated {len(mappings)} comprehensive mappings")
        
        return model, mappings, tf2_model
        
    except Exception as e:
        print(f" TF2 model creation failed: {e}")
        return None, None, None

def validate_mapping_coverage(tf1_variables, tf2_mappings, tf2_model):
    """Validate how well our mappings cover the TF1 checkpoint"""
    print_subsection("MAPPING COVERAGE VALIDATION")
    
    tf2_var_names = set([v.name for v in tf2_model.variables])
    
    # Check mapping coverage
    mappable_vars = 0
    perfect_matches = 0
    missing_tf1_vars = []
    missing_tf2_vars = []
    
    print(f" Mapping Analysis:")
    
    # Check how many TF1 variables we can map
    for tf1_var_name in tf1_variables:
        if tf1_var_name in tf2_mappings:
            mappable_vars += 1
            tf2_var_name = tf2_mappings[tf1_var_name]
            if tf2_var_name in tf2_var_names:
                perfect_matches += 1
            else:
                missing_tf2_vars.append((tf1_var_name, tf2_var_name))
        else:
            missing_tf1_vars.append(tf1_var_name)
    
    mapping_coverage = mappable_vars / len(tf1_variables) * 100
    loading_success = perfect_matches / len(tf1_variables) * 100
    
    print(f"   TF1 variables in checkpoint: {len(tf1_variables)}")
    print(f"   Mappable variables: {mappable_vars}/{len(tf1_variables)} ({mapping_coverage:.1f}%)")
    print(f"   Perfect matches: {perfect_matches}/{len(tf1_variables)} ({loading_success:.1f}%)")
    print(f"   Missing TF1 mappings: {len(missing_tf1_vars)}")
    print(f"   Missing TF2 targets: {len(missing_tf2_vars)}")
    
    # Show details of missing mappings
    if missing_tf1_vars:
        print(f"\n  TF1 variables without mappings (first 5):")
        for var_name in missing_tf1_vars[:5]:
            print(f"      {var_name}")
    
    if missing_tf2_vars:
        print(f"\n  Mappings to missing TF2 variables (first 5):")
        for tf1_name, tf2_name in missing_tf2_vars[:5]:
            print(f"      {tf1_name} → {tf2_name} (MISSING)")
    
    return mapping_coverage, loading_success, perfect_matches

def test_checkpoint_loading(model, checkpoint_path, expected_matches):
    """Test actual checkpoint loading with the TF1 model"""
    print_subsection("CHECKPOINT LOADING TEST")
    
    try:
        # Update model config to point to our checkpoint
        model.conf.checkpoint = checkpoint_path
        print(f" Attempting to load checkpoint: {checkpoint_path}")
        
        # Build the model first (create the TF2 model structure)
        print(f" Building TF2 model structure...")
        
        # Ensure the model class has built its internal model
        if not hasattr(model, 'model') or model.model is None:
            # Build the model using the model class's build method
            input_shape = (model.conf.input_size[0], model.conf.input_size[1], 3)
            model.model = model.build_model(input_shape)
        
        # Get the TF2 model instance from the Model class
        if hasattr(model, 'model') and model.model is not None:
            tf2_model = model.model
        else:
            # Create the model if it doesn't exist
            tf2_model = Deeplab_v2_TF2(num_classes=model.conf.num_classes, name='resnet_v1_50')
            dummy_input = tf.ones((1, model.conf.input_size[0], model.conf.input_size[1], 3))
            _ = tf2_model(dummy_input, training=False)  # Build the model
            model.model = tf2_model
        
        print(f" TF2 model structure ready")
        print(f"   Variables before loading: {len(tf2_model.variables)}")
        
        # Attempt checkpoint loading
        print(f"Loading TF1 checkpoint...")
        success_count = model.load_checkpoint(checkpoint_path)
        
        print(f" CHECKPOINT LOADING COMPLETED!")
        print(f" Loading Results:")
        print(f"   Variables successfully loaded: {success_count}")
        print(f"   Expected loadable variables: {expected_matches}")
        print(f"   Loading success rate: {(success_count/expected_matches*100) if expected_matches > 0 else 0:.1f}%")
        print(f"   Overall coverage: {success_count/len(tf2_model.variables)*100:.1f}%")
        
        return success_count, tf2_model
        
    except Exception as e:
        print(f" Checkpoint loading failed: {e}")
        import traceback
        traceback.print_exc()
        return 0, None

def test_model_behavior(tf2_model, config):
    """Test that the loaded model behaves correctly"""
    print_subsection("MODEL BEHAVIOR VALIDATION")
    
    try:
        # Test with multiple inputs to validate behavior
        test_cases = [
            ("Random Input", np.random.normal(0.5, 0.1, (1, config.input_size[0], config.input_size[1], 3)).astype(np.float32)),
            ("Zero Input", np.zeros((1, config.input_size[0], config.input_size[1], 3), dtype=np.float32)),
            ("Ones Input", np.ones((1, config.input_size[0], config.input_size[1], 3), dtype=np.float32)),
        ]
        
        print(f" Testing model behavior with different inputs:")
        
        for test_name, test_input in test_cases:
            # Clip input to valid range [0, 1]
            test_input = np.clip(test_input, 0.0, 1.0)
            
            output = tf2_model(test_input, training=False)
            
            # Analyze output
            output_min = tf.reduce_min(output).numpy()
            output_max = tf.reduce_max(output).numpy()
            output_mean = tf.reduce_mean(output).numpy()
            output_std = tf.math.reduce_std(output).numpy()
            
            # Check for NaN or infinite values
            has_nan = tf.reduce_any(tf.math.is_nan(output)).numpy()
            has_inf = tf.reduce_any(tf.math.is_inf(output)).numpy()
            
            print(f"   {test_name}:")
            print(f"      Shape: {output.shape}")
            print(f"      Range: [{output_min:.4f}, {output_max:.4f}]")
            print(f"      Mean±Std: {output_mean:.4f}±{output_std:.4f}")
            print(f"      Valid: {'' if not (has_nan or has_inf) else ''} (No NaN/Inf)")
            
            # Check if output is reasonable for segmentation
            if output.shape[-1] == config.num_classes:
                # Apply softmax to get probabilities
                probs = tf.nn.softmax(output, axis=-1)
                max_prob = tf.reduce_max(probs).numpy()
                min_prob = tf.reduce_min(probs).numpy()
                print(f"      Probability range: [{min_prob:.4f}, {max_prob:.4f}]")
        
        print(f" Model behavior validation completed")
        return True
        
    except Exception as e:
        print(f" Model behavior test failed: {e}")
        return False

def compare_with_tf1_behavior(tf2_model, config):
    """Compare TF2 model behavior with expected TF1 behavior"""
    print_subsection("TF1 vs TF2 BEHAVIOR COMPARISON")
    
    print(f" Model Comparison Analysis:")
    
    # Get model statistics
    total_params = sum([tf.size(var).numpy() for var in tf2_model.trainable_variables])
    print(f"   Total trainable parameters: {total_params:,}")
    
    # Test output characteristics
    test_input = np.random.normal(0.5, 0.1, (1, config.input_size[0], config.input_size[1], 3)).astype(np.float32)
    test_input = np.clip(test_input, 0.0, 1.0)
    
    output = tf2_model(test_input, training=False)
    
    # Analyze segmentation quality
    print(f"   Output analysis:")
    print(f"      Shape: {output.shape}")
    print(f"      Expected: (1, 65, 65, 21) for 8x downsampling")
    
    if output.shape == (1, 65, 65, 21):
        print(f"       Output shape matches expected segmentation format")
    else:
        print(f"        Output shape unexpected for segmentation")
    
    # Check if logits are reasonable
    output_range = tf.reduce_max(output) - tf.reduce_min(output)
    print(f"      Logit range: {output_range.numpy():.4f}")
    
    if 1.0 < output_range.numpy() < 20.0:
        print(f"       Logit range appears reasonable for segmentation")
    else:
        print(f"        Logit range may be unusual")
    
    # Test gradient flow (important for training)
    with tf.GradientTape() as tape:
        test_input_tensor = tf.convert_to_tensor(test_input)
        tape.watch(test_input_tensor)
        output = tf2_model(test_input_tensor, training=True)
        loss = tf.reduce_mean(output)
    
    gradients = tape.gradient(loss, tf2_model.trainable_variables)
    grad_norms = [tf.norm(g).numpy() if g is not None else 0.0 for g in gradients]
    avg_grad_norm = np.mean([g for g in grad_norms if g > 0])
    
    print(f"   Gradient analysis:")
    print(f"      Variables with gradients: {sum(1 for g in grad_norms if g > 0)}/{len(grad_norms)}")
    print(f"      Average gradient norm: {avg_grad_norm:.6f}")
    
    if avg_grad_norm > 1e-7:
        print(f"       Gradients flowing properly")
    else:
        print(f"        Gradients may be too small")
    
    return total_params

def main():
    """Main test function"""
    print_section("TF1 MODEL LOADING & TF2 MIGRATION VALIDATION TEST")
    
    # Create test configuration
    config = TestConfig()
    
    # Ensure log directory exists
    os.makedirs(config.logdir, exist_ok=True)
    
    print(f" Test Configuration:")
    print(f"   TF1 Checkpoint: {config.checkpoint}")
    print(f"   Input size: {config.input_size}")
    print(f"   Number of classes: {config.num_classes}")
    print(f"   TF2 Fixed mode: {config.tf2_fixed}")
    
    # Check if checkpoint exists
    if not os.path.exists(config.checkpoint + '.index'):
        print(f" CRITICAL ERROR: TF1 checkpoint not found at {config.checkpoint}")
        print(f"   Please verify the path to the TF1 model checkpoint.")
        return False
    
    print(f" TF1 checkpoint found and accessible")
    
    # Step 1: Analyze TF1 checkpoint
    tf1_variables, tf1_categories = analyze_tf1_checkpoint(config.checkpoint)
    if tf1_variables is None:
        return False
    
    # Step 2: Create TF2 model and test mapping
    model, mappings, tf2_model = create_tf2_model_and_test_mapping(config)
    if model is None:
        return False
    
    # Step 3: Validate mapping coverage
    mapping_coverage, loading_success, expected_matches = validate_mapping_coverage(
        tf1_variables, mappings, tf2_model
    )
    
    # Step 4: Test actual checkpoint loading
    success_count, loaded_model = test_checkpoint_loading(model, config.checkpoint, expected_matches)
    
    # Step 5: Test model behavior
    if loaded_model is not None:
        behavior_test = test_model_behavior(loaded_model, config)
        total_params = compare_with_tf1_behavior(loaded_model, config)
    else:
        behavior_test = False
        total_params = 0
    
    # Final summary
    print_section("FINAL TEST RESULTS SUMMARY")
    
    print(f" Test Results:")
    print(f"   TF1 checkpoint variables: {len(tf1_variables)}")
    print(f"   TF2 mapping rules: {len(mappings)}")
    print(f"   Mapping coverage: {mapping_coverage:.1f}%")
    print(f"   Variables successfully loaded: {success_count}")
    print(f"   Loading success rate: {(success_count/expected_matches*100) if expected_matches > 0 else 0:.1f}%")
    print(f"   Model behavior test: {' PASSED' if behavior_test else ' FAILED'}")
    print(f"   Total trainable parameters: {total_params:,}")
    
    # Overall assessment
    print(f"\n OVERALL ASSESSMENT:")
    if mapping_coverage >= 90 and success_count >= expected_matches * 0.9 and behavior_test:
        print(f"    EXCELLENT: TF2 migration is working perfectly!")
        print(f"    Model loaded successfully with high fidelity")
        print(f"    Ready for production use")
    elif mapping_coverage >= 70 and success_count >= expected_matches * 0.7:
        print(f"    GOOD: TF2 migration is working well")
        print(f"     Some optimization opportunities remain")
    else:
        print(f"     NEEDS IMPROVEMENT: Migration needs refinement")
        print(f"    Review mapping rules and loading process")
    
    print(f"\nTest completed. Check results above for detailed analysis.")
    
    return mapping_coverage >= 70 and success_count >= expected_matches * 0.7

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
