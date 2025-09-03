import tensorflow as tf
import sys
import six

class ResNet_segmentation_TF2(tf.keras.Model):
    """
    TF2 native implementation of ResNet segmentation network.
    This replaces the TF1 ResNet_segmentation class with proper TF2 structure.
    """
    
    def __init__(self, num_classes, encoder_name='res50', **kwargs):
        super().__init__(**kwargs)
        
        if encoder_name not in ['res101', 'res50']:
            print('encoder_name ERROR!')
            print("Please input: res101, res50")
            sys.exit(-1)
            
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.channel_axis = 3
        
        # Build the network structure
        self._build_network_structure()
    
    def _build_network_structure(self):
        """Build the network layer structure without calling layers yet."""
        
        # Initial convolution block - EXACT TF1 NAMING for checkpoint compatibility
        self.conv1 = tf.keras.layers.Conv2D(
            64, 7, strides=2, padding='same', 
            use_bias=False, name='conv1'
        )
        self.bn_conv1 = tf.keras.layers.BatchNormalization(
            axis=3, name='bn_conv1'
        )
        
        # Max pooling
        self.max_pool = tf.keras.layers.MaxPooling2D(
            pool_size=3, strides=2, padding='same'
        )
        
        # ResNet blocks - build structure to match TF1 naming exactly
        self._build_resnet_blocks()
        
        # ASPP decoder layers - EXACT TF1 NAMING
        self._build_aspp_layers()
    
    def _build_resnet_blocks(self):
        """Build ResNet bottleneck blocks with exact TF1 naming."""
        
        # Block 1 (3 units) - 256 channels
        self.block1_units = []
        self.block1_units.append(self._create_bottleneck_unit(64, 256, 'unit_1', identity_connection=False))
        self.block1_units.append(self._create_bottleneck_unit(256, 256, 'unit_2'))
        self.block1_units.append(self._create_bottleneck_unit(256, 256, 'unit_3'))
        
        # Block 2 (4 units) - 512 channels
        self.block2_units = []
        self.block2_units.append(self._create_bottleneck_unit(256, 512, 'unit_1', half_size=True, identity_connection=False))
        for i in range(2, 5):
            self.block2_units.append(self._create_bottleneck_unit(512, 512, f'unit_{i}'))
        
        # Block 3 (variable units) - 1024 channels with dilation=2
        self.block3_units = []
        num_layers_block3 = 23 if self.encoder_name == 'res101' else 6
        self.block3_units.append(self._create_dilated_bottleneck_unit(512, 1024, 2, 'unit_1', identity_connection=False))
        for i in range(2, num_layers_block3 + 1):
            self.block3_units.append(self._create_dilated_bottleneck_unit(1024, 1024, 2, f'unit_{i}'))
        
        # Block 4 (3 units) - 2048 channels with dilation=4  
        self.block4_units = []
        self.block4_units.append(self._create_dilated_bottleneck_unit(1024, 2048, 4, 'unit_1', identity_connection=False))
        self.block4_units.append(self._create_dilated_bottleneck_unit(2048, 2048, 4, 'unit_2'))
        self.block4_units.append(self._create_dilated_bottleneck_unit(2048, 2048, 4, 'unit_3'))
    
    def _create_bottleneck_unit(self, input_channels, output_channels, unit_name, 
                               half_size=False, identity_connection=True):
        """Create a ResNet bottleneck unit with exact TF1 naming."""
        
        strides = 2 if half_size else 1
        bottleneck_channels = output_channels // 4
        
        # Branch 2a (1x1 conv)
        conv2a = tf.keras.layers.Conv2D(
            bottleneck_channels, 1, strides=strides, padding='same',
            use_bias=False, name=f'res{unit_name}_branch2a'
        )
        bn2a = tf.keras.layers.BatchNormalization(
            axis=3, name=f'bn{unit_name}_branch2a'
        )
        
        # Branch 2b (3x3 conv)  
        conv2b = tf.keras.layers.Conv2D(
            bottleneck_channels, 3, strides=1, padding='same',
            use_bias=False, name=f'res{unit_name}_branch2b'
        )
        bn2b = tf.keras.layers.BatchNormalization(
            axis=3, name=f'bn{unit_name}_branch2b'
        )
        
        # Branch 2c (1x1 conv)
        conv2c = tf.keras.layers.Conv2D(
            output_channels, 1, strides=1, padding='same',
            use_bias=False, name=f'res{unit_name}_branch2c'
        )
        bn2c = tf.keras.layers.BatchNormalization(
            axis=3, name=f'bn{unit_name}_branch2c'
        )
        
        # Branch 1 (shortcut) - only if not identity connection
        branch1_layers = None
        if not identity_connection:
            conv1 = tf.keras.layers.Conv2D(
                output_channels, 1, strides=strides, padding='same',
                use_bias=False, name=f'res{unit_name}_branch1'
            )
            bn1 = tf.keras.layers.BatchNormalization(
                axis=3, name=f'bn{unit_name}_branch1'
            )
            branch1_layers = (conv1, bn1)
        
        return {
            'branch2a': (conv2a, bn2a),
            'branch2b': (conv2b, bn2b),
            'branch2c': (conv2c, bn2c),
            'branch1': branch1_layers,
            'identity_connection': identity_connection
        }
    
    def _create_dilated_bottleneck_unit(self, input_channels, output_channels, dilation_rate, 
                                       unit_name, identity_connection=True):
        """Create a dilated ResNet bottleneck unit with exact TF1 naming."""
        
        bottleneck_channels = output_channels // 4
        
        # Branch 2a (1x1 conv)
        conv2a = tf.keras.layers.Conv2D(
            bottleneck_channels, 1, strides=1, padding='same',
            use_bias=False, name=f'res{unit_name}_branch2a'
        )
        bn2a = tf.keras.layers.BatchNormalization(
            axis=3, name=f'bn{unit_name}_branch2a'
        )
        
        # Branch 2b (3x3 dilated conv)
        conv2b = tf.keras.layers.Conv2D(
            bottleneck_channels, 3, strides=1, padding='same',
            dilation_rate=dilation_rate, use_bias=False, 
            name=f'res{unit_name}_branch2b'
        )
        bn2b = tf.keras.layers.BatchNormalization(
            axis=3, name=f'bn{unit_name}_branch2b'
        )
        
        # Branch 2c (1x1 conv)
        conv2c = tf.keras.layers.Conv2D(
            output_channels, 1, strides=1, padding='same',
            use_bias=False, name=f'res{unit_name}_branch2c'
        )
        bn2c = tf.keras.layers.BatchNormalization(
            axis=3, name=f'bn{unit_name}_branch2c'
        )
        
        # Branch 1 (shortcut) - only if not identity connection
        branch1_layers = None
        if not identity_connection:
            conv1 = tf.keras.layers.Conv2D(
                output_channels, 1, strides=1, padding='same',
                use_bias=False, name=f'res{unit_name}_branch1'
            )
            bn1 = tf.keras.layers.BatchNormalization(
                axis=3, name=f'bn{unit_name}_branch1'
            )
            branch1_layers = (conv1, bn1)
        
        return {
            'branch2a': (conv2a, bn2a),
            'branch2b': (conv2b, bn2b),
            'branch2c': (conv2c, bn2c),
            'branch1': branch1_layers,
            'identity_connection': identity_connection,
            'dilation_rate': dilation_rate
        }
    
    def _build_aspp_layers(self):
        """Build ASPP layers with exact TF1 naming for checkpoint compatibility."""
        
        # ASPP branches with different dilation rates
        self.fc1_voc12_c0 = tf.keras.layers.Conv2D(
            self.num_classes, 3, padding='same', dilation_rate=6,
            use_bias=True, name='fc1_voc12_c0'
        )
        
        self.fc1_voc12_c1 = tf.keras.layers.Conv2D(
            self.num_classes, 3, padding='same', dilation_rate=12,
            use_bias=True, name='fc1_voc12_c1'
        )
        
        self.fc1_voc12_c2 = tf.keras.layers.Conv2D(
            self.num_classes, 3, padding='same', dilation_rate=18,
            use_bias=True, name='fc1_voc12_c2'
        )
        
        self.fc1_voc12_c3 = tf.keras.layers.Conv2D(
            self.num_classes, 3, padding='same', dilation_rate=24,
            use_bias=True, name='fc1_voc12_c3'
        )
    
    def call(self, inputs, training=None):
        """Forward pass through the network."""
        
        print("Creating ResNet segmentation model in TF2 NATIVE mode")
        print(f"-----------build encoder: {self.encoder_name}-----------")
        
        # Initial convolution block
        x = self.conv1(inputs)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)
        print("after start block:", x.shape)
        
        # Block 1
        for i, unit in enumerate(self.block1_units):
            x = self._apply_bottleneck_unit(x, unit, training=training)
        print("after block1:", x.shape)
        
        # Block 2
        for i, unit in enumerate(self.block2_units):
            x = self._apply_bottleneck_unit(x, unit, training=training)
        print("after block2:", x.shape)
        
        # Block 3 (dilated)
        for i, unit in enumerate(self.block3_units):
            x = self._apply_bottleneck_unit(x, unit, training=training)
        print("after block3:", x.shape)
        
        # Block 4 (dilated)
        for i, unit in enumerate(self.block4_units):
            x = self._apply_bottleneck_unit(x, unit, training=training)
        print("after block4:", x.shape)
        
        # ASPP decoder
        print("-----------build decoder-----------")
        aspp_outputs = []
        aspp_outputs.append(self.fc1_voc12_c0(x))
        aspp_outputs.append(self.fc1_voc12_c1(x))
        aspp_outputs.append(self.fc1_voc12_c2(x))
        aspp_outputs.append(self.fc1_voc12_c3(x))
        
        # Add ASPP outputs
        x = tf.add_n(aspp_outputs, name='fc1_voc12')
        print("after ASPP block:", x.shape)
        
        return x
    
    def _apply_bottleneck_unit(self, x, unit, training=None):
        """Apply a bottleneck unit to input tensor."""
        
        # Branch 2a
        conv2a, bn2a = unit['branch2a']
        branch2 = conv2a(x)
        branch2 = bn2a(branch2, training=training)
        branch2 = tf.nn.relu(branch2)
        
        # Branch 2b
        conv2b, bn2b = unit['branch2b']
        branch2 = conv2b(branch2)
        branch2 = bn2b(branch2, training=training)
        branch2 = tf.nn.relu(branch2)
        
        # Branch 2c
        conv2c, bn2c = unit['branch2c']
        branch2 = conv2c(branch2)
        branch2 = bn2c(branch2, training=training)
        
        # Branch 1 (shortcut)
        if unit['identity_connection']:
            branch1 = x
        else:
            conv1, bn1 = unit['branch1']
            branch1 = conv1(x)
            branch1 = bn1(branch1, training=training)
        
        # Add branches and apply ReLU
        output = tf.add(branch1, branch2)
        output = tf.nn.relu(output)
        
        return output


class Deeplab_v2_TF2_FIXED(tf.keras.Model):
    """
    FIXED TF2 implementation of DeepLab v2 that properly builds variables.
    This replaces the problematic TF1-style implementation.
    """
    
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
        # Use the proper TF2 ResNet implementation
        self.resnet = ResNet_segmentation_TF2(
            num_classes=num_classes, 
            encoder_name='res50',
            name='resnet_v1_50'
        )
    
    def call(self, inputs, training=None):
        """Forward pass through DeepLab v2."""
        return self.resnet(inputs, training=training)
    
    def get_variables_summary(self):
        """Get a summary of all variables for debugging."""
        variables = self.trainable_variables
        print(f"\nTF2 MODEL VARIABLES SUMMARY:")
        print(f"Total trainable variables: {len(variables)}")
        
        if len(variables) > 0:
            print("\nFirst 10 variables:")
            for i, var in enumerate(variables[:10]):
                print(f"  {i+1:2d}. {var.name:50s} {str(var.shape):20s}")
        
        return variables
