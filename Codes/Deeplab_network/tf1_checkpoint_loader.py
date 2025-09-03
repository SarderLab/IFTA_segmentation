"""
TF1 Checkpoint Loader for DeepLab ResNet-50 Model
Handles loading TF1 checkpoint weights into TF2 models with comprehensive variable mapping.
"""

import os
import glob
import tensorflow as tf
import numpy as np


class TF1CheckpointLoader:
    """
    Specialized class for loading TF1 DeepLab ResNet-50 checkpoints into TF2 models.
    
    This class handles:
    - TF1 checkpoint reading and validation
    - Comprehensive variable name mapping (TF1 -> TF2)
    - Smart variable assignment with shape validation
    - Fuzzy matching for unmapped variables
    - Loading progress tracking and reporting
    """
    
    def __init__(self, tf2_model):
        """
        Initialize the TF1 checkpoint loader.
        
        Args:
            tf2_model: The TF2 model to load weights into
        """
        self.tf2_model = tf2_model
        self.mapping_rules = self._create_comprehensive_mapping_rules()
        
    def load_checkpoint(self, checkpoint_path):
        """
        Load TF1 checkpoint weights into the TF2 model.
        
        Args:
            checkpoint_path: Path to the TF1 checkpoint (e.g., 'model.ckpt-1')
            
        Returns:
            int: Number of variables successfully loaded
            
        Raises:
            FileNotFoundError: If checkpoint files are not found
            RuntimeError: If no variables could be loaded
        """
        # Validate and locate checkpoint files
        tf1_checkpoint_path = self._validate_checkpoint_path(checkpoint_path)
        
        # Read TF1 checkpoint
        print(f"Loading TF1 checkpoint: {tf1_checkpoint_path}")
        reader = tf.train.load_checkpoint(tf1_checkpoint_path)
        var_map = reader.get_variable_to_shape_map()
        print(f"TF1 checkpoint contains {len(var_map)} variables")
        
        # Ensure TF2 model is built
        self._ensure_model_built()
        
        # Load variables with mapping
        loaded_count = self._load_variables_with_mapping(reader, var_map)
        
        # Validate loading results
        if loaded_count == 0:
            raise RuntimeError("CRITICAL: No variables were loaded from checkpoint!")
            
        print(f"\nCHECKPOINT LOADING SUMMARY:")
        print(f"  Successfully loaded: {loaded_count} variables")
        print(f"  Success rate: {loaded_count/len(var_map)*100:.1f}%")
        print(f"TF1 checkpoint loading completed!")
        
        return loaded_count
    
    def _validate_checkpoint_path(self, checkpoint_path):
        """Validate and normalize checkpoint path."""
        # Handle checkpoint path - if it already looks like a TF1 checkpoint, use it directly
        if checkpoint_path.endswith('.ckpt-1') or '.ckpt-' in checkpoint_path:
            tf1_checkpoint_pattern = checkpoint_path
        else:
            # Original logic for other formats
            model_dir = os.path.dirname(checkpoint_path)
            expected_step = os.path.basename(checkpoint_path).split('_')[-1]
            tf1_checkpoint_pattern = os.path.join(model_dir, f'model.ckpt-{expected_step}')
        
        # Check for TF1 format checkpoint files
        tf1_files = glob.glob(tf1_checkpoint_pattern + '*')
        
        if not tf1_files:
            raise FileNotFoundError(f"No TF1 checkpoint found at {tf1_checkpoint_pattern}")
            
        return tf1_checkpoint_pattern
    
    def _ensure_model_built(self):
        """Ensure the TF2 model is properly built with enhanced strategies."""
        if not self.tf2_model.built:
            print("Building TF2 model structure...")
            dummy_input = tf.random.normal((1, 256, 256, 3))
            _ = self.tf2_model(dummy_input, training=False)
        
        # CRITICAL: Force model to actually create variables by calling it multiple times if needed
        if len(self.tf2_model.trainable_variables) == 0:
            print("WARNING: Model has 0 variables - forcing model building with enhanced strategies...")
            dummy_input = tf.random.normal((1, 256, 256, 3))
            try:
                # Multiple attempts to build the model with different strategies
                for i in range(3):
                    output = self.tf2_model(dummy_input, training=False)
                    if len(self.tf2_model.trainable_variables) > 0:
                        break
                    print(f"   Attempt {i+1}: Still 0 variables, trying training=True...")
                    output = self.tf2_model(dummy_input, training=True)
                    if len(self.tf2_model.trainable_variables) > 0:
                        break
                    print(f"   Attempt {i+1}: Trying different input sizes...")
                    dummy_input = tf.random.normal((1, 513, 513, 3))
                    output = self.tf2_model(dummy_input, training=False)
                    if len(self.tf2_model.trainable_variables) > 0:
                        break
            except Exception as e:
                print(f"   Model building error: {e}")
                # Try to build manually if the model has a build method
                if hasattr(self.tf2_model, 'build'):
                    try:
                        self.tf2_model.build((None, 256, 256, 3))
                    except Exception as e2:
                        print(f"   Manual build failed: {e2}")
            
        print(f"TF2 model has {len(self.tf2_model.trainable_variables)} trainable variables")
        print(f"TF2 model has {len(self.tf2_model.variables)} total variables")
        
        # Display actual TF2 variable names for debugging
        print(f"\nACTUAL TF2 VARIABLE NAMES (first 20):")
        for i, var in enumerate(self.tf2_model.variables[:20]):
            print(f"  {i+1:2d}. {var.name:<50} {tuple(var.shape)}")
        if len(self.tf2_model.variables) > 20:
            print(f"     ... and {len(self.tf2_model.variables) - 20} more")
    
    def _load_variables_with_mapping(self, reader, var_map):
        """Load variables using comprehensive mapping rules."""
        loaded_count = 0
        skipped_count = 0
        
        print(f"COMPREHENSIVE MAPPINGS: {len(self.mapping_rules)} total rules created")
        
        for tf1_var_name, tf1_shape in var_map.items():
            # Skip momentum and other optimizer variables for inference
            if '/Momentum' in tf1_var_name or '/Adam' in tf1_var_name or '/RMSProp' in tf1_var_name:
                continue
                
            try:
                tf1_value = reader.get_tensor(tf1_var_name)
                
                # Apply mapping rules
                tf2_var = self._map_and_assign_variable(tf1_var_name, tf1_value)
                
                if tf2_var is not None:
                    loaded_count += 1
                    if loaded_count <= 30:  # Show first 30 successful mappings
                        print(f"MAPPED: {tf1_var_name} -> {tf2_var.name}")
                else:
                    skipped_count += 1
                    if skipped_count <= 15:  # Show first 15 failures
                        print(f"WARNING: No mapping found for: {tf1_var_name}")
                        
            except Exception as e:
                print(f"ERROR loading {tf1_var_name}: {e}")
                skipped_count += 1
        
        return loaded_count
    
    def _map_and_assign_variable(self, tf1_var_name, tf1_value):
        """Map TF1 variable to TF2 and assign value."""
        # Try direct mapping first
        if tf1_var_name in self.mapping_rules:
            tf2_var_name = self.mapping_rules[tf1_var_name]
            result = self._assign_variable_by_name(tf2_var_name, tf1_value)
            if result is not None:
                return result
        
        # Enhanced fuzzy matching with better scoring
        return self._fuzzy_map_variable(tf1_var_name, tf1_value)
    
    def _assign_variable_by_name(self, tf2_var_name, tf1_value):
        """Assign TF1 value to TF2 variable by name."""
        # Find the TF2 variable by name
        for var in self.tf2_model.variables:
            if var.name == tf2_var_name:
                # Check shape compatibility
                if var.shape.as_list() == list(tf1_value.shape):
                    var.assign(tf1_value)
                    return var
                else:
                    print(f"   WARNING: Shape mismatch: {tf1_value.shape} -> {var.shape}")
                    return None
        return None
    
    def _fuzzy_map_variable(self, tf1_var_name, tf1_value):
        """Enhanced fuzzy matching for unmapped variables with comprehensive strategies."""
        # Extract meaningful components from TF1 variable name
        tf1_clean = tf1_var_name.replace('/weights', '').replace('/biases', '')
        tf1_clean = tf1_clean.replace('/gamma', '').replace('/beta', '')
        tf1_clean = tf1_clean.replace('/moving_mean', '').replace('/moving_variance', '')
        tf1_clean = tf1_clean.replace('/BatchNorm', '')
        
        tf1_parts = set(tf1_clean.split('/'))
        
        # Determine variable type
        var_type = None
        if '/weights' in tf1_var_name:
            var_type = 'kernel'
        elif '/biases' in tf1_var_name:
            var_type = 'bias'
        elif '/gamma' in tf1_var_name:
            var_type = 'gamma'
        elif '/beta' in tf1_var_name:
            var_type = 'beta'
        elif '/moving_mean' in tf1_var_name:
            var_type = 'moving_mean'
        elif '/moving_variance' in tf1_var_name:
            var_type = 'moving_variance'
        
        # Find best matching TF2 variable using multiple strategies
        best_score = 0
        best_var = None
        
        # Strategy 1: Standard fuzzy matching
        for var in self.tf2_model.variables:
            if var.shape.as_list() != list(tf1_value.shape):
                continue
                
            # Check variable type compatibility
            if var_type and var_type not in var.name:
                continue
                
            # Score based on name component overlap
            tf2_parts = set(var.name.replace(':0', '').split('/'))
            overlap = len(tf1_parts.intersection(tf2_parts))
            score = overlap / max(len(tf1_parts), len(tf2_parts))
            
            if score > best_score:
                best_score = score
                best_var = var
        
        # Strategy 2: Enhanced pattern matching with TF1->TF2 common patterns
        if best_var is None or best_score < 0.5:
            enhanced_patterns = [
                # Direct mappings
                (tf1_var_name, tf1_var_name.replace('/weights', '/kernel:0').replace('/biases', '/bias:0')),
                
                # BatchNorm mappings
                (tf1_var_name, tf1_var_name.replace('/gamma', '/gamma:0').replace('/beta', '/beta:0')),
                (tf1_var_name, tf1_var_name.replace('/moving_mean', '/moving_mean:0').replace('/moving_variance', '/moving_variance:0')),
                
                # ResNet block patterns with different prefixes
                (tf1_var_name, f'resnet_wrapper/{tf1_var_name.replace("/weights", "/kernel:0")}'),
                (tf1_var_name, f'deeplab_v2/{tf1_var_name.replace("/weights", "/kernel:0")}'),
                (tf1_var_name, f'deeplab_wrapper/{tf1_var_name.replace("/weights", "/kernel:0")}'),
                
                # Handle res blocks with bottleneck structure
                (tf1_var_name, tf1_var_name.replace('res', 'bottleneck_').replace('/weights', '/kernel:0')),
                (tf1_var_name, f'resnet_wrapper/bottleneck_{tf1_var_name[3:]}'.replace('/weights', '/kernel:0') if tf1_var_name.startswith('res') else None),
                
                # Handle batch norm with bn prefix
                (tf1_var_name, tf1_var_name.replace('bn', 'batch_normalization').replace('/gamma', '/gamma:0').replace('/beta', '/beta:0')),
                
                # ASPP layer mappings
                (tf1_var_name, tf1_var_name.replace('/weights', '/kernel:0').replace('/biases', '/bias:0') if 'fc1_voc12' in tf1_var_name else None),
                
                # Alternative model wrapper names
                (tf1_var_name, f'model/{tf1_var_name.replace("/weights", "/kernel:0")}'),
                (tf1_var_name, f'functional_1/{tf1_var_name.replace("/weights", "/kernel:0")}'),
                (tf1_var_name, f'resnet_v1_50/{tf1_var_name.replace("/weights", "/kernel:0")}'),
            ]
            
            for original_name, mapped_name in enhanced_patterns:
                if mapped_name is None:
                    continue
                    
                for var in self.tf2_model.variables:
                    if var.name == mapped_name and var.shape.as_list() == list(tf1_value.shape):
                        best_var = var
                        best_score = 1.0
                        break
                
                if best_score >= 1.0:
                    break
        
        # Strategy 3: Substring and component matching with lower threshold
        if best_var is None or best_score < 0.3:
            for var in self.tf2_model.variables:
                if var.shape.as_list() != list(tf1_value.shape):
                    continue
                
                # Extract components from TF2 variable name
                tf2_clean = var.name.replace('/kernel:0', '').replace('/bias:0', '')
                tf2_clean = tf2_clean.replace('/gamma:0', '').replace('/beta:0', '')
                tf2_clean = tf2_clean.replace('/moving_mean:0', '').replace('/moving_variance:0', '')
                
                tf2_parts = set(tf2_clean.split('/'))
                
                # Calculate similarity score
                common = tf1_parts & tf2_parts
                union = tf1_parts | tf2_parts
                score = len(common) / len(union) if union else 0
                
                # Bonus for exact substring matches
                for tf1_part in tf1_parts:
                    for tf2_part in tf2_parts:
                        if tf1_part in tf2_part or tf2_part in tf1_part:
                            score += 0.1
                
                if score > best_score and score >= 0.15:  # Lower threshold for maximum coverage
                    best_score = score
                    best_var = var
        
        # Only assign if we have a reasonable match
        if best_var is not None and best_score > 0.2:
            best_var.assign(tf1_value)
            return best_var
            
        return None
    
    def _create_comprehensive_mapping_rules(self):
        """Create COMPREHENSIVE mapping rules for TF1 to TF2 variable names - ENHANCED VERSION."""
        mappings = {}
        
        # 1. INITIAL CONVOLUTION (5 variables) - ENHANCED WITH MULTIPLE PATTERNS
        initial_mappings = {
            # Original patterns
            'conv1/weights': 'resnet_v1_50/conv1/kernel:0',
            'bn_conv1/gamma': 'resnet_v1_50/bn_conv1/gamma:0',
            'bn_conv1/beta': 'resnet_v1_50/bn_conv1/beta:0',
            'bn_conv1/moving_mean': 'resnet_v1_50/bn_conv1/moving_mean:0',
            'bn_conv1/moving_variance': 'resnet_v1_50/bn_conv1/moving_variance:0',
            
            # Double-wrapped patterns from model.py
            'conv1/weights': 'resnet_v1_50/resnet_v1_50/conv1/kernel:0',
            'bn_conv1/gamma': 'resnet_v1_50/resnet_v1_50/bn_conv1/gamma:0',
            'bn_conv1/beta': 'resnet_v1_50/resnet_v1_50/bn_conv1/beta:0',
            'bn_conv1/moving_mean': 'resnet_v1_50/resnet_v1_50/bn_conv1/moving_mean:0',
            'bn_conv1/moving_variance': 'resnet_v1_50/resnet_v1_50/bn_conv1/moving_variance:0',
        }
        mappings.update(initial_mappings)
        
        # 2. ASPP LAYERS (8 variables) - ENHANCED WITH MULTIPLE PATTERNS
        aspp_mappings = {
            # Original patterns
            'fc1_voc12_c0/weights': 'resnet_v1_50/fc1_voc12_c0/kernel:0',
            'fc1_voc12_c0/biases': 'resnet_v1_50/fc1_voc12_c0/bias:0',
            'fc1_voc12_c1/weights': 'resnet_v1_50/fc1_voc12_c1/kernel:0',
            'fc1_voc12_c1/biases': 'resnet_v1_50/fc1_voc12_c1/bias:0',
            'fc1_voc12_c2/weights': 'resnet_v1_50/fc1_voc12_c2/kernel:0',
            'fc1_voc12_c2/biases': 'resnet_v1_50/fc1_voc12_c2/bias:0',
            'fc1_voc12_c3/weights': 'resnet_v1_50/fc1_voc12_c3/kernel:0',
            'fc1_voc12_c3/biases': 'resnet_v1_50/fc1_voc12_c3/bias:0',
            
            # Double-wrapped patterns from model.py
            'fc1_voc12_c0/weights': 'resnet_v1_50/resnet_v1_50/fc1_voc12_c0/kernel:0',
            'fc1_voc12_c0/biases': 'resnet_v1_50/resnet_v1_50/fc1_voc12_c0/bias:0',
            'fc1_voc12_c1/weights': 'resnet_v1_50/resnet_v1_50/fc1_voc12_c1/kernel:0',
            'fc1_voc12_c1/biases': 'resnet_v1_50/resnet_v1_50/fc1_voc12_c1/bias:0',
            'fc1_voc12_c2/weights': 'resnet_v1_50/resnet_v1_50/fc1_voc12_c2/kernel:0',
            'fc1_voc12_c2/biases': 'resnet_v1_50/resnet_v1_50/fc1_voc12_c2/bias:0',
            'fc1_voc12_c3/weights': 'resnet_v1_50/resnet_v1_50/fc1_voc12_c3/kernel:0',
            'fc1_voc12_c3/biases': 'resnet_v1_50/resnet_v1_50/fc1_voc12_c3/bias:0',
        }
        mappings.update(aspp_mappings)
        
        # 3. ENHANCED RESNET BLOCKS MAPPING - Multiple patterns for compatibility
        # Pattern 1: Standard ResNet units (original approach)
        resnet_units_standard = [
            # Block 2
            ('res2a', 'bottleneck_2a'),
            ('res2b', 'bottleneck_2b'), 
            ('res2c', 'bottleneck_2c'),
            # Block 3  
            ('res3a', 'bottleneck_3a'),
            ('res3b1', 'bottleneck_3b1'),
            ('res3b2', 'bottleneck_3b2'),
            ('res3b3', 'bottleneck_3b3'),
            # Block 4
            ('res4a', 'bottleneck_4a'),
        ]
        
        # Add res4b1-22 (dilated bottlenecks)
        for i in range(1, 23):
            resnet_units_standard.append((f'res4b{i}', f'dilated_bottleneck_4b{i}'))
        
        # Block 5 (dilated)
        resnet_units_standard.extend([
            ('res5a', 'dilated_bottleneck_5a'),
            ('res5b', 'dilated_bottleneck_5b'),
            ('res5c', 'dilated_bottleneck_5c'),
        ])
        
        # Pattern 2: Corrected ResNet mapping from model.py (unit-based)
        tf1_to_tf2_unit_mapping = {
            'res2a': 'unit_1',  # First unit with shortcut
            'res2b': 'unit_2',  # No shortcut
            'res2c': 'unit_3',  # No shortcut
            'res3a': 'unit_4',  # No shortcut
            'res3b1': 'unit_5', # No shortcut
            'res3b2': 'unit_6', # No shortcut
        }
        
        # Generate mappings for standard pattern
        for tf1_unit, tf2_unit in resnet_units_standard:
            branches = ['1', '2a', '2b', '2c']
            for branch in branches:
                # Weights mapping
                mappings[f'{tf1_unit}_branch{branch}/weights'] = f'resnet_v1_50/{tf2_unit}/{tf1_unit}_branch{branch}/kernel:0'
                # BatchNorm mappings
                mappings[f'bn{tf1_unit}_branch{branch}/gamma'] = f'resnet_v1_50/{tf2_unit}/bn{tf1_unit}_branch{branch}/gamma:0'
                mappings[f'bn{tf1_unit}_branch{branch}/beta'] = f'resnet_v1_50/{tf2_unit}/bn{tf1_unit}_branch{branch}/beta:0'
                mappings[f'bn{tf1_unit}_branch{branch}/moving_mean'] = f'resnet_v1_50/{tf2_unit}/bn{tf1_unit}_branch{branch}/moving_mean:0'
                mappings[f'bn{tf1_unit}_branch{branch}/moving_variance'] = f'resnet_v1_50/{tf2_unit}/bn{tf1_unit}_branch{branch}/moving_variance:0'
        
        # Generate mappings for corrected unit-based pattern
        for tf1_unit, tf2_unit in tf1_to_tf2_unit_mapping.items():
            # Branch 1 (shortcut) - ONLY for unit_1
            if tf2_unit == 'unit_1':
                mappings[f'{tf1_unit}_branch1/weights'] = f'resnet_v1_50/resnet_v1_50/res{tf2_unit}_branch1/kernel:0'
                mappings[f'bn{tf1_unit}_branch1/gamma'] = f'resnet_v1_50/resnet_v1_50/bn{tf2_unit}_branch1/gamma:0'
                mappings[f'bn{tf1_unit}_branch1/beta'] = f'resnet_v1_50/resnet_v1_50/bn{tf2_unit}_branch1/beta:0'
                mappings[f'bn{tf1_unit}_branch1/moving_mean'] = f'resnet_v1_50/resnet_v1_50/bn{tf2_unit}_branch1/moving_mean:0'
                mappings[f'bn{tf1_unit}_branch1/moving_variance'] = f'resnet_v1_50/resnet_v1_50/bn{tf2_unit}_branch1/moving_variance:0'
            
            # Branch 2a, 2b, 2c (standard bottleneck branches) - ALL units have these
            for branch in ['2a', '2b', '2c']:
                mappings[f'{tf1_unit}_branch{branch}/weights'] = f'resnet_v1_50/resnet_v1_50/res{tf2_unit}_branch{branch}/kernel:0'
                mappings[f'bn{tf1_unit}_branch{branch}/gamma'] = f'resnet_v1_50/resnet_v1_50/bn{tf2_unit}_branch{branch}/gamma:0'
                mappings[f'bn{tf1_unit}_branch{branch}/beta'] = f'resnet_v1_50/resnet_v1_50/bn{tf2_unit}_branch{branch}/beta:0'
                mappings[f'bn{tf1_unit}_branch{branch}/moving_mean'] = f'resnet_v1_50/resnet_v1_50/bn{tf2_unit}_branch{branch}/moving_mean:0'
                mappings[f'bn{tf1_unit}_branch{branch}/moving_variance'] = f'resnet_v1_50/resnet_v1_50/bn{tf2_unit}_branch{branch}/moving_variance:0'
        
        print(f"ENHANCED COMPREHENSIVE MAPPINGS: {len(mappings)} total rules created")
        print(f"    - Initial conv: {len([k for k in mappings.keys() if 'conv1' in k or 'bn_conv1' in k])} variables")
        print(f"    - ASPP layers: {len([k for k in mappings.keys() if 'fc1_voc12' in k])} variables")
        print(f"    - ResNet blocks: {len(mappings) - len([k for k in mappings.keys() if 'conv1' in k or 'bn_conv1' in k or 'fc1_voc12' in k])} variables")
        
        return mappings
    
    def analyze_checkpoint(self, checkpoint_path):
        """
        Analyze TF1 checkpoint structure without loading.
        
        Args:
            checkpoint_path: Path to the TF1 checkpoint
            
        Returns:
            dict: Analysis results with variable categories and statistics
        """
        tf1_checkpoint_path = self._validate_checkpoint_path(checkpoint_path)
        
        # Read checkpoint metadata
        reader = tf.train.load_checkpoint(tf1_checkpoint_path)
        var_map = reader.get_variable_to_shape_map()
        
        # Categorize variables
        categories = {
            'Initial Conv': [],
            'ASPP': [],
            'ResNet Blocks': [],
            'Other': []
        }
        
        for var_name, shape in var_map.items():
            if 'conv1' in var_name or 'bn_conv1' in var_name:
                categories['Initial Conv'].append((var_name, shape))
            elif 'fc1_voc12' in var_name:
                categories['ASPP'].append((var_name, shape))
            elif any(block in var_name for block in ['res2', 'res3', 'res4', 'res5']):
                categories['ResNet Blocks'].append((var_name, shape))
            else:
                categories['Other'].append((var_name, shape))
        
        return {
            'total_variables': len(var_map),
            'categories': categories,
            'mappable_count': len([v for v in var_map.keys() if v in self.mapping_rules]),
            'mapping_coverage': len([v for v in var_map.keys() if v in self.mapping_rules]) / len(var_map) * 100
        }
    
    def get_mapping_statistics(self):
        """Get statistics about the mapping rules."""
        return {
            'total_mappings': len(self.mapping_rules),
            'initial_conv_mappings': len([k for k in self.mapping_rules.keys() if 'conv1' in k or 'bn_conv1' in k]),
            'aspp_mappings': len([k for k in self.mapping_rules.keys() if 'fc1_voc12' in k]),
            'resnet_mappings': len(self.mapping_rules) - 13,
        }
