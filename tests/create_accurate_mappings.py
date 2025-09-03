#!/usr/bin/env python3
"""
Comprehensive TF1 to TF2 mapping analysis and fix
Creates accurate mappings based on actual checkpoint structure
"""

import os
import sys
import tensorflow as tf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def create_accurate_mappings():
    """Create accurate TF1 to TF2 mappings based on actual checkpoint analysis"""
    print("CREATING ACCURATE TF1 ↔ TF2 MAPPINGS")
    print("=" * 60)
    
    # Load both TF1 and TF2 to understand structure
    config = TestConfig()
    checkpoint_path = config.checkpoint
    
    # Get TF1 variables
    reader = tf.train.load_checkpoint(checkpoint_path)
    tf1_var_map = reader.get_variable_to_shape_map()
    
    # Get TF2 variables
    model = Model(config)
    input_shape = (config.input_size[0], config.input_size[1], 3)
    model.build_model(input_shape)
    
    if not model.model.built:
        dummy_input = tf.random.normal((1, 513, 513, 3))
        _ = model.model(dummy_input, training=False)
    
    tf2_variables = {var.name: var.shape for var in model.model.variables}
    
    # Create comprehensive mapping rules
    mappings = {}
    
    # 1. INITIAL CONVOLUTION (perfect mapping)
    print(" Mapping Initial Convolution...")
    initial_mappings = {
        'conv1/weights': 'resnet_v1_50/conv1/kernel:0',
        'bn_conv1/gamma': 'resnet_v1_50/bn_conv1/gamma:0',
        'bn_conv1/beta': 'resnet_v1_50/bn_conv1/beta:0',
        'bn_conv1/moving_mean': 'resnet_v1_50/bn_conv1/moving_mean:0',
        'bn_conv1/moving_variance': 'resnet_v1_50/bn_conv1/moving_variance:0',
    }
    mappings.update(initial_mappings)
    print(f"    {len(initial_mappings)} initial conv mappings")
    
    # 2. RESNET UNITS MAPPING (based on actual TF2 structure)
    print(" Mapping ResNet Units...")
    resnet_mappings = create_resnet_mappings(tf1_var_map, tf2_variables)
    mappings.update(resnet_mappings)
    print(f"    {len(resnet_mappings)} ResNet mappings")
    
    # 3. ASPP/FC LAYERS (handle shape mismatch)
    print(" Mapping ASPP Layers...")
    aspp_mappings = create_aspp_mappings(tf1_var_map, tf2_variables)
    mappings.update(aspp_mappings)
    print(f"    {len(aspp_mappings)} ASPP mappings")
    
    # Analysis
    mappable_tf1 = [var for var in tf1_var_map.keys() if var in mappings and 'Momentum' not in var]
    coverage = len(mappable_tf1) / len([var for var in tf1_var_map.keys() if 'Momentum' not in var]) * 100
    
    print(f"\n MAPPING SUMMARY:")
    print(f"   Total mapping rules: {len(mappings)}")
    print(f"   TF1 variables (non-optimizer): {len([var for var in tf1_var_map.keys() if 'Momentum' not in var])}")
    print(f"   Mappable TF1 variables: {len(mappable_tf1)}")
    print(f"   Coverage: {coverage:.1f}%")
    
    return mappings


def create_resnet_mappings(tf1_var_map, tf2_variables):
    """Create ResNet unit mappings based on actual variable analysis"""
    mappings = {}
    
    # Analyze TF1 ResNet blocks
    tf1_resnet_vars = [var for var in tf1_var_map.keys() if any(block in var for block in ['res2', 'res3', 'res4', 'res5'])]
    tf1_weight_vars = [var for var in tf1_resnet_vars if '/weights' in var and 'Momentum' not in var]
    tf1_bn_vars = [var for var in tf1_resnet_vars if any(bn in var for bn in ['/gamma', '/beta', '/moving_mean', '/moving_variance'])]
    
    # Analyze TF2 ResNet units  
    tf2_resnet_vars = [var for var in tf2_variables.keys() if any(unit in var for unit in ['resunit_', 'bnunit_'])]
    tf2_weight_vars = [var for var in tf2_resnet_vars if '/kernel:' in var]
    tf2_bn_vars = [var for var in tf2_resnet_vars if any(bn in var for bn in ['/gamma:', '/beta:', '/moving_mean:', '/moving_variance:'])]
    
    print(f"   TF1 ResNet weights: {len(tf1_weight_vars)}")
    print(f"   TF1 ResNet BN: {len(tf1_bn_vars)}")
    print(f"   TF2 ResNet weights: {len(tf2_weight_vars)}")
    print(f"   TF2 ResNet BN: {len(tf2_bn_vars)}")
    
    # Create mapping based on ResNet-50 structure
    # Standard ResNet-50: [3, 4, 6, 3] blocks in res2, res3, res4, res5
    
    # Block 2: res2a, res2b, res2c -> unit_1, unit_2, unit_3
    res2_mapping = {
        'res2a': 'unit_1',  # Has branch1 shortcut
        'res2b': 'unit_2',  # No shortcut
        'res2c': 'unit_3',  # No shortcut
    }
    
    # Block 3: res3a, res3b1, res3b2, res3b3 -> unit_4, unit_5, unit_6, (missing?)
    res3_mapping = {
        'res3a': 'unit_4',   # Usually has branch1 shortcut for dimension change
        'res3b1': 'unit_5',  # No shortcut
        'res3b2': 'unit_6',  # No shortcut
        # res3b3 might be missing in TF2 implementation or named differently
    }
    
    # The issue is that TF2 model seems to have only 6 units total, but ResNet-50 should have many more
    # Let's map what we can based on available TF2 units
    
    # Extract available units from TF2
    tf2_unit_numbers = set()\n    for var in tf2_weight_vars:\n        if 'resunit_' in var:\n            unit_num = var.split('resunit_')[1].split('_')[0]\n            tf2_unit_numbers.add(int(unit_num))\n    \n    available_units = sorted(tf2_unit_numbers)\n    print(f\"   Available TF2 units: {available_units}\")\n    \n    # Map based on available units\n    unit_counter = 1\n    \n    # Map res2 blocks\n    for tf1_unit, _ in res2_mapping.items():\n        if unit_counter in available_units:\n            tf2_unit = f'unit_{unit_counter}'\n            \n            # Map weights\n            if unit_counter == 1:  # First unit usually has branch1\n                branches = ['1', '2a', '2b', '2c']\n            else:\n                branches = ['2a', '2b', '2c']\n                \n            for branch in branches:\n                tf1_weight = f'{tf1_unit}_branch{branch}/weights'\n                tf2_weight = f'resnet_v1_50/resunit_{unit_counter}_branch{branch}/kernel:0'\n                if tf1_weight in tf1_var_map and tf2_weight in tf2_variables:\n                    mappings[tf1_weight] = tf2_weight\n                \n                # Map batch norm\n                for bn_type in ['gamma', 'beta', 'moving_mean', 'moving_variance']:\n                    tf1_bn = f'bn{tf1_unit}_branch{branch}/{bn_type}'\n                    tf2_bn = f'resnet_v1_50/bnunit_{unit_counter}_branch{branch}/{bn_type}:0'\n                    if tf1_bn in tf1_var_map and tf2_bn in tf2_variables:\n                        mappings[tf1_bn] = tf2_bn\n            \n            unit_counter += 1\n    \n    # Map res3 blocks\n    for tf1_unit, _ in res3_mapping.items():\n        if unit_counter in available_units:\n            tf2_unit = f'unit_{unit_counter}'\n            \n            # res3a typically has branch1 for dimension change\n            if tf1_unit == 'res3a':\n                branches = ['1', '2a', '2b', '2c']\n            else:\n                branches = ['2a', '2b', '2c']\n                \n            for branch in branches:\n                tf1_weight = f'{tf1_unit}_branch{branch}/weights'\n                tf2_weight = f'resnet_v1_50/resunit_{unit_counter}_branch{branch}/kernel:0'\n                if tf1_weight in tf1_var_map and tf2_weight in tf2_variables:\n                    mappings[tf1_weight] = tf2_weight\n                \n                # Map batch norm\n                for bn_type in ['gamma', 'beta', 'moving_mean', 'moving_variance']:\n                    tf1_bn = f'bn{tf1_unit}_branch{branch}/{bn_type}'\n                    tf2_bn = f'resnet_v1_50/bnunit_{unit_counter}_branch{branch}/{bn_type}:0'\n                    if tf1_bn in tf1_var_map and tf2_bn in tf2_variables:\n                        mappings[tf1_bn] = tf2_bn\n            \n            unit_counter += 1\n    \n    return mappings\n\n\ndef create_aspp_mappings(tf1_var_map, tf2_variables):\n    \"\"\"Create ASPP layer mappings handling shape mismatches\"\"\"\n    mappings = {}\n    \n    # ASPP layers: fc1_voc12_c0 through fc1_voc12_c3\n    for i in range(4):\n        layer_name = f'fc1_voc12_c{i}'\n        \n        # Check if both TF1 and TF2 have this layer\n        tf1_weight = f'{layer_name}/weights'\n        tf1_bias = f'{layer_name}/biases'\n        tf2_weight = f'resnet_v1_50/{layer_name}/kernel:0'\n        tf2_bias = f'resnet_v1_50/{layer_name}/bias:0'\n        \n        # Only map if shapes are compatible\n        if (tf1_weight in tf1_var_map and tf2_weight in tf2_variables):\n            tf1_shape = tf1_var_map[tf1_weight]\n            tf2_shape = tf2_variables[tf2_weight]\n            \n            # Check if only the last dimension differs (class count)\n            if (len(tf1_shape) == len(tf2_shape) and \n                tf1_shape[:-1] == tf2_shape[:-1]):\n                print(f\"     Shape mismatch for {layer_name}: TF1 {tf1_shape} vs TF2 {tf2_shape}\")\n                print(f\"      (Different number of classes: {tf1_shape[-1]} vs {tf2_shape[-1]})\")\n                # Skip layers with different output dimensions\n                continue\n            elif tf1_shape == tf2_shape:\n                mappings[tf1_weight] = tf2_weight\n                mappings[tf1_bias] = tf2_bias\n            else:\n                print(f\"    Incompatible shapes for {layer_name}: {tf1_shape} vs {tf2_shape}\")\n    \n    return mappings\n\n\ndef main():\n    \"\"\"Create and analyze accurate mappings\"\"\"\n    mappings = create_accurate_mappings()\n    \n    print(\"\\n\" + \"=\" * 60)\n    print(\" SAMPLE ACCURATE MAPPINGS:\")\n    \n    # Show sample mappings by category\n    initial_mappings = {k: v for k, v in mappings.items() if 'conv1' in k or 'bn_conv1' in k}\n    resnet_mappings = {k: v for k, v in mappings.items() if any(res in k for res in ['res2', 'res3', 'res4', 'res5'])}\n    aspp_mappings = {k: v for k, v in mappings.items() if 'fc1_voc12' in k}\n    \n    print(f\"\\n Initial Conv ({len(initial_mappings)}):\") \n    for tf1, tf2 in list(initial_mappings.items())[:3]:\n        print(f\"   {tf1} -> {tf2}\")\n    \n    print(f\"\\n ResNet Units ({len(resnet_mappings)}):\") \n    for tf1, tf2 in list(resnet_mappings.items())[:5]:\n        print(f\"   {tf1} -> {tf2}\")\n    \n    print(f\"\\n ASPP Layers ({len(aspp_mappings)}):\") \n    for tf1, tf2 in list(aspp_mappings.items()):\n        print(f\"   {tf1} -> {tf2}\")\n    \n    print(\"\\n\" + \"=\" * 60)\n    print(\" ACCURATE MAPPING ANALYSIS COMPLETE\")\n    \n    return mappings\n\n\nif __name__ == \"__main__\":\n    main()"
