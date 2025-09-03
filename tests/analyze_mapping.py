#!/usr/bin/env python3
"""
Detailed analysis of TF1 checkpoint structure and TF2 model variables
This will help understand the 18% mapping coverage issue
"""

import os
import sys
import tensorflow as tf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Codes.Deeplab_network.tf1_checkpoint_loader import TF1CheckpointLoader
from Codes.Deeplab_network.model import Model


class TestConfig:
    """Configuration for analysis"""
    def __init__(self):
        self.num_classes = 21
        self.ignore_label = 255
        self.logdir = '/tmp/analysis_logs'
        self.learning_rate = 0.0001
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.bsize = 1
        self.input_size = (513, 513)
        self.tf2_fixed = True
        self.encoder_name = 'res50'
        self.num_steps = 1000
        self.save_interval = 100
        self.power = 0.9
        self.print_color = '\033[92m'
        self.logfile = '/tmp/analysis.log'
        self.modeldir = '/tmp/analysis_models'
        self.valid_step = 1000
        self.valid_num_steps = 10
        self.test_step = 1000
        self.test_data_list = '/tmp/test_list.txt'
        self.out_dir = '/tmp/test_output'
        self.visual = False
        self.test_num_steps = 10
        self.batch_size = 1
        self.checkpoint = '/home/anish.tatke/blue-group/anish.tatke/IFTA_segmentation/DiagnosticCore/MODELS/0/HR/model.ckpt-1'


def analyze_tf1_checkpoint():
    """Analyze the TF1 checkpoint structure"""
    config = TestConfig()
    checkpoint_path = config.checkpoint
    
    print("ANALYZING TF1 CHECKPOINT STRUCTURE")
    print("=" * 60)
    
    if not os.path.exists(checkpoint_path + '.index'):
        print(f" TF1 checkpoint not found: {checkpoint_path}")
        return None
        
    # Read TF1 checkpoint
    reader = tf.train.load_checkpoint(checkpoint_path)
    var_map = reader.get_variable_to_shape_map()
    
    print(f" TF1 Checkpoint contains {len(var_map)} variables")
    
    # Categorize variables
    categories = {
        'Initial Conv': [],
        'ResNet Block 2': [],
        'ResNet Block 3': [],
        'ResNet Block 4': [],
        'ResNet Block 5': [],
        'ASPP/FC Layers': [],
        'Optimizer Variables': [],
        'Other': []
    }
    
    for var_name, shape in var_map.items():
        if 'conv1' in var_name or 'bn_conv1' in var_name:
            categories['Initial Conv'].append((var_name, shape))
        elif 'res2' in var_name:
            categories['ResNet Block 2'].append((var_name, shape))
        elif 'res3' in var_name:
            categories['ResNet Block 3'].append((var_name, shape))
        elif 'res4' in var_name:
            categories['ResNet Block 4'].append((var_name, shape))
        elif 'res5' in var_name:
            categories['ResNet Block 5'].append((var_name, shape))
        elif 'fc1' in var_name or 'fc_' in var_name:
            categories['ASPP/FC Layers'].append((var_name, shape))
        elif 'Momentum' in var_name or 'Adam' in var_name or 'RMSProp' in var_name:
            categories['Optimizer Variables'].append((var_name, shape))
        else:
            categories['Other'].append((var_name, shape))
    
    # Print category breakdown
    for category, variables in categories.items():
        print(f"\n {category}: {len(variables)} variables")
        if len(variables) > 0:
            print(f"   Sample variables:")
            for i, (name, shape) in enumerate(variables[:5]):
                print(f"     {i+1}. {name:<50} {shape}")
            if len(variables) > 5:
                print(f"     ... and {len(variables) - 5} more")
    
    return var_map


def analyze_tf2_model():
    """Analyze the TF2 model structure"""
    print("\n\nANALYZING TF2 MODEL STRUCTURE")
    print("=" * 60)
    
    config = TestConfig()
    model = Model(config)
    
    # Build the model
    input_shape = (config.input_size[0], config.input_size[1], 3)
    model.build_model(input_shape)
    
    # Ensure model is built
    if not model.model.built:
        dummy_input = tf.random.normal((1, 513, 513, 3))
        _ = model.model(dummy_input, training=False)
    
    print(f" TF2 Model contains:")
    print(f"   - Trainable variables: {len(model.model.trainable_variables)}")
    print(f"   - Total variables: {len(model.model.variables)}")
    
    # Categorize TF2 variables
    tf2_categories = {
        'Initial Conv': [],
        'ResNet Units': [],
        'ASPP/FC Layers': [],
        'Batch Norm': [],
        'Other': []
    }
    
    for var in model.model.variables:
        var_name = var.name
        if 'conv1' in var_name and 'resnet_v1_50' in var_name:
            tf2_categories['Initial Conv'].append((var_name, var.shape))
        elif any(unit in var_name for unit in ['unit_', 'bottleneck_']):
            tf2_categories['ResNet Units'].append((var_name, var.shape))
        elif 'fc1' in var_name or 'fc_' in var_name:
            tf2_categories['ASPP/FC Layers'].append((var_name, var.shape))
        elif any(bn in var_name for bn in ['batch_normalization', '/gamma:', '/beta:', '/moving_']):
            tf2_categories['Batch Norm'].append((var_name, var.shape))
        else:
            tf2_categories['Other'].append((var_name, var.shape))
    
    # Print TF2 category breakdown
    for category, variables in tf2_categories.items():
        print(f"\n {category}: {len(variables)} variables")
        if len(variables) > 0:
            print(f"   Sample variables:")
            for i, (name, shape) in enumerate(variables[:5]):
                print(f"     {i+1}. {name:<60} {shape}")
            if len(variables) > 5:
                print(f"     ... and {len(variables) - 5} more")
    
    # Print ALL TF2 variable names for detailed analysis
    print(f"\n ALL TF2 VARIABLE NAMES:")
    for i, var in enumerate(model.model.variables):
        print(f"  {i+1:3d}. {var.name}")
    
    return model


def analyze_mapping_coverage():
    """Analyze mapping coverage between TF1 and TF2"""
    print("\n\nANALYZING MAPPING COVERAGE")
    print("=" * 60)
    
    config = TestConfig()
    
    # Create model and loader
    model = Model(config)
    input_shape = (config.input_size[0], config.input_size[1], 3)
    model.build_model(input_shape)
    
    # Build model
    if not model.model.built:
        dummy_input = tf.random.normal((1, 513, 513, 3))
        _ = model.model(dummy_input, training=False)
    
    loader = TF1CheckpointLoader(model.model)
    
    # Analyze checkpoint if available
    checkpoint_path = config.checkpoint
    if os.path.exists(checkpoint_path + '.index'):
        analysis = loader.analyze_checkpoint(checkpoint_path)
        
        print(f" MAPPING COVERAGE ANALYSIS:")
        print(f"   Total TF1 variables: {analysis['total_variables']}")
        print(f"   Mappable variables: {analysis['mappable_count']}")
        print(f"   Coverage percentage: {analysis['mapping_coverage']:.1f}%")
        
        print(f"\n TF1 VARIABLE CATEGORIES:")
        for category, variables in analysis['categories'].items():
            mapped_count = len([v for v, _ in variables if v in loader.mapping_rules])
            print(f"   {category}: {len(variables)} total, {mapped_count} mapped ({mapped_count/len(variables)*100:.1f}%)")
        
        # Show unmapped variables
        reader = tf.train.load_checkpoint(checkpoint_path)
        var_map = reader.get_variable_to_shape_map()
        unmapped = [var for var in var_map.keys() if var not in loader.mapping_rules and 'Momentum' not in var and 'Adam' not in var]
        
        print(f"\n UNMAPPED VARIABLES ({len(unmapped)}):")
        for i, var_name in enumerate(unmapped[:20]):  # Show first 20
            print(f"   {i+1:2d}. {var_name}")
        if len(unmapped) > 20:
            print(f"      ... and {len(unmapped) - 20} more")
            
        return analysis
    else:
        print(" TF1 checkpoint not found for analysis")
        return None


def main():
    """Run comprehensive analysis"""
    print("COMPREHENSIVE TF1 ↔ TF2 MAPPING ANALYSIS")
    print("=" * 80)
    
    # Analyze TF1 checkpoint structure
    tf1_vars = analyze_tf1_checkpoint()
    
    # Analyze TF2 model structure
    tf2_model = analyze_tf2_model()
    
    # Analyze mapping coverage
    coverage_analysis = analyze_mapping_coverage()
    
    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETE")
    
    if coverage_analysis:
        print(f" Current mapping coverage: {coverage_analysis['mapping_coverage']:.1f}%")
        print(" Review the unmapped variables above to improve coverage")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
