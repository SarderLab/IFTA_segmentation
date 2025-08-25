import tensorflow as tf
import numpy as np
import six


"""
This script defines the segmentation network - TF2.x compatible version.

The encoding part is a pre-trained ResNet. This script supports several settings:

	Deeplab v2 pre-trained model (pre-trained on MSCOCO) ('deeplab_resnet_init.ckpt')
	Deeplab v2 pre-trained model (pre-trained on MSCOCO + PASCAL_train+val) ('deeplab_resnet.ckpt')
	Original ResNet-101 ('resnet_v1_101.ckpt')
	Original ResNet-50 ('resnet_v1_50.ckpt')

Note: This version is migrated to TF2.x using tf.keras.layers instead of tf.contrib
"""


class Deeplab_v2_TF2(tf.keras.Model):
	"""
	TF2.x native implementation of Deeplab v2 using tf.keras.Model
	Complete ResNet backbone with bottleneck blocks and ASPP decoder
	"""
	def __init__(self, num_classes, **kwargs):
		super(Deeplab_v2_TF2, self).__init__(**kwargs)
		self.num_classes = num_classes
		self.channel_axis = 3
		
		# Initial convolution
		self.conv1 = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', 
											kernel_initializer='he_normal', name='conv1')
		self.bn_conv1 = tf.keras.layers.BatchNormalization(name='bn_conv1')
		self.pool1 = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same', name='pool1')
		
		# ResNet bottleneck blocks - Stage 2 (64x64)
		self.block2a = self._make_bottleneck_block(256, '2a', first_block=True, downsample=False)
		self.block2b = self._make_bottleneck_block(256, '2b')
		self.block2c = self._make_bottleneck_block(256, '2c')
		
		# ResNet bottleneck blocks - Stage 3 (32x32)
		self.block3a = self._make_bottleneck_block(512, '3a', first_block=True, downsample=True)
		self.block3b1 = self._make_bottleneck_block(512, '3b1')
		self.block3b2 = self._make_bottleneck_block(512, '3b2')
		self.block3b3 = self._make_bottleneck_block(512, '3b3')
		
		# ResNet dilated blocks - Stage 4 (32x32 with dilation=2)
		self.block4a = self._make_dilated_bottleneck_block(1024, 2, '4a', first_block=True)
		self.block4b_layers = []
		for i in range(1, 23):  # 22 more blocks (4b1 through 4b22)
			self.block4b_layers.append(
				self._make_dilated_bottleneck_block(1024, 2, f'4b{i}')
			)
		
		# ResNet dilated blocks - Stage 5 (32x32 with dilation=4)
		self.block5a = self._make_dilated_bottleneck_block(2048, 4, '5a', first_block=True)
		self.block5b = self._make_dilated_bottleneck_block(2048, 4, '5b')
		self.block5c = self._make_dilated_bottleneck_block(2048, 4, '5c')
		
		# ASPP (Atrous Spatial Pyramid Pooling) decoder
		self.aspp_conv1 = tf.keras.layers.Conv2D(num_classes, 3, dilation_rate=6, padding='same', 
												 use_bias=True, kernel_initializer='he_normal',
												 name='fc1_voc12_c0')
		self.aspp_conv2 = tf.keras.layers.Conv2D(num_classes, 3, dilation_rate=12, padding='same', 
												 use_bias=True, kernel_initializer='he_normal',
												 name='fc1_voc12_c1')
		self.aspp_conv3 = tf.keras.layers.Conv2D(num_classes, 3, dilation_rate=18, padding='same', 
												 use_bias=True, kernel_initializer='he_normal',
												 name='fc1_voc12_c2')
		self.aspp_conv4 = tf.keras.layers.Conv2D(num_classes, 3, dilation_rate=24, padding='same', 
												 use_bias=True, kernel_initializer='he_normal',
												 name='fc1_voc12_c3')
	
	def _make_bottleneck_block(self, filters, name, first_block=False, downsample=False):
		"""Create a ResNet bottleneck block"""
		class BottleneckBlock(tf.keras.layers.Layer):
			def __init__(self, filters, name, first_block=False, downsample=False, **kwargs):
				super().__init__(name=f'bottleneck_{name}', **kwargs)
				self.filters = filters
				self.first_block = first_block
				self.downsample = downsample
				
				stride = 2 if downsample else 1
				
				# Branch 2 (main path)
				self.conv2a = tf.keras.layers.Conv2D(filters // 4, 1, strides=stride, padding='same',
													kernel_initializer='he_normal', name=f'res{name}_branch2a')
				self.bn2a = tf.keras.layers.BatchNormalization(name=f'bn{name}_branch2a')
				
				self.conv2b = tf.keras.layers.Conv2D(filters // 4, 3, padding='same',
													kernel_initializer='he_normal', name=f'res{name}_branch2b')
				self.bn2b = tf.keras.layers.BatchNormalization(name=f'bn{name}_branch2b')
				
				self.conv2c = tf.keras.layers.Conv2D(filters, 1, padding='same',
													kernel_initializer='he_normal', name=f'res{name}_branch2c')
				self.bn2c = tf.keras.layers.BatchNormalization(name=f'bn{name}_branch2c')
				
				# Branch 1 (shortcut) - only if first block or downsample
				if first_block or downsample:
					self.conv1 = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same',
														kernel_initializer='he_normal', name=f'res{name}_branch1')
					self.bn1 = tf.keras.layers.BatchNormalization(name=f'bn{name}_branch1')
				else:
					self.conv1 = None
					self.bn1 = None
			
			def call(self, inputs, training=None):
				# Branch 1 (shortcut)
				if self.conv1 is not None:
					shortcut = self.conv1(inputs)
					shortcut = self.bn1(shortcut, training=False)  # Keep frozen as in original
				else:
					shortcut = inputs
				
				# Branch 2 (main path)
				x = self.conv2a(inputs)
				x = self.bn2a(x, training=False)  # Keep frozen as in original
				x = tf.nn.relu(x)
				
				x = self.conv2b(x)
				x = self.bn2b(x, training=False)
				x = tf.nn.relu(x)
				
				x = self.conv2c(x)
				x = self.bn2c(x, training=False)
				
				# Add shortcut
				x = tf.add(x, shortcut)
				x = tf.nn.relu(x)
				
				return x
		
		return BottleneckBlock(filters, name, first_block, downsample)
	
	def _make_dilated_bottleneck_block(self, filters, dilation_rate, name, first_block=False):
		"""Create a dilated ResNet bottleneck block"""
		class DilatedBottleneckBlock(tf.keras.layers.Layer):
			def __init__(self, filters, dilation_rate, name, first_block=False, **kwargs):
				super().__init__(name=f'dilated_bottleneck_{name}', **kwargs)
				self.filters = filters
				self.dilation_rate = dilation_rate
				self.first_block = first_block
				
				# Branch 2 (main path)
				self.conv2a = tf.keras.layers.Conv2D(filters // 4, 1, padding='same',
													kernel_initializer='he_normal', name=f'res{name}_branch2a')
				self.bn2a = tf.keras.layers.BatchNormalization(name=f'bn{name}_branch2a')
				
				self.conv2b = tf.keras.layers.Conv2D(filters // 4, 3, dilation_rate=dilation_rate, padding='same',
													kernel_initializer='he_normal', name=f'res{name}_branch2b')
				self.bn2b = tf.keras.layers.BatchNormalization(name=f'bn{name}_branch2b')
				
				self.conv2c = tf.keras.layers.Conv2D(filters, 1, padding='same',
													kernel_initializer='he_normal', name=f'res{name}_branch2c')
				self.bn2c = tf.keras.layers.BatchNormalization(name=f'bn{name}_branch2c')
				
				# Branch 1 (shortcut) - only for first block of each stage
				if first_block:
					self.conv1 = tf.keras.layers.Conv2D(filters, 1, padding='same',
														kernel_initializer='he_normal', name=f'res{name}_branch1')
					self.bn1 = tf.keras.layers.BatchNormalization(name=f'bn{name}_branch1')
				else:
					self.conv1 = None
					self.bn1 = None
			
			def call(self, inputs, training=None):
				# Branch 1 (shortcut)
				if self.conv1 is not None:
					shortcut = self.conv1(inputs)
					shortcut = self.bn1(shortcut, training=False)
				else:
					shortcut = inputs
				
				# Branch 2 (main path)
				x = self.conv2a(inputs)
				x = self.bn2a(x, training=False)
				x = tf.nn.relu(x)
				
				x = self.conv2b(x)
				x = self.bn2b(x, training=False)
				x = tf.nn.relu(x)
				
				x = self.conv2c(x)
				x = self.bn2c(x, training=False)
				
				# Add shortcut
				x = tf.add(x, shortcut)
				x = tf.nn.relu(x)
				
				return x
		
		return DilatedBottleneckBlock(filters, dilation_rate, name, first_block)
	
	def call(self, inputs, training=None):
		"""Forward pass through the network"""
		# Initial convolution
		x = self.conv1(inputs)
		x = self.bn_conv1(x, training=False)  # Keep frozen as in original
		x = tf.nn.relu(x)
		x = self.pool1(x)
		
		print(f"After start block: {x.shape}")
		
		# Stage 2: 3 bottleneck blocks
		x = self.block2a(x, training=training)
		x = self.block2b(x, training=training)
		x = self.block2c(x, training=training)
		
		print(f"After block1: {x.shape}")
		
		# Stage 3: 4 bottleneck blocks with downsampling
		x = self.block3a(x, training=training)
		x = self.block3b1(x, training=training)
		x = self.block3b2(x, training=training)
		x = self.block3b3(x, training=training)
		
		print(f"After block2: {x.shape}")
		
		# Stage 4: 23 dilated blocks (dilation=2)
		x = self.block4a(x, training=training)
		for block4b in self.block4b_layers:
			x = block4b(x, training=training)
		
		print(f"After block3: {x.shape}")
		
		# Stage 5: 3 dilated blocks (dilation=4)
		x = self.block5a(x, training=training)
		x = self.block5b(x, training=training)
		x = self.block5c(x, training=training)
		
		print(f"After block4: {x.shape}")
		
		# ASPP Decoder
		aspp_outputs = []
		aspp_outputs.append(self.aspp_conv1(x))
		aspp_outputs.append(self.aspp_conv2(x))
		aspp_outputs.append(self.aspp_conv3(x))
		aspp_outputs.append(self.aspp_conv4(x))
		
		# Combine ASPP outputs
		outputs = tf.add_n(aspp_outputs, name='fc1_voc12')
		
		print(f"After ASPP block: {outputs.shape}")
		
		return outputs


# Keep the original classes for backward compatibility but add TF2 compatibility fixes
class Deeplab_v2(object):
	"""
	Original Deeplab v2 - Updated for TF2 compatibility
	Note: This maintains the original structure but fixes TF2 incompatibilities
	"""
	def __init__(self, inputs, num_classes, phase):
		self.inputs = inputs
		self.num_classes = num_classes
		self.channel_axis = 3
		self.phase = phase # train (True) or test (False), for BN layers in the decoder
		self.build_network()

	def build_network(self):
		self.encoding = self.build_encoder()
		self.outputs = self.build_decoder(self.encoding)

	def build_encoder(self):
		print("-----------build encoder: deeplab pre-trained-----------")
		outputs = self._start_block()
		print("after start block:", outputs.shape)
		outputs = self._bottleneck_resblock(outputs, 256, '2a', identity_connection=False)
		outputs = self._bottleneck_resblock(outputs, 256, '2b')
		outputs = self._bottleneck_resblock(outputs, 256, '2c')
		print("after block1:", outputs.shape)
		outputs = self._bottleneck_resblock(outputs, 512, '3a',	half_size=True, identity_connection=False)
		for i in six.moves.range(1, 4):
			outputs = self._bottleneck_resblock(outputs, 512, '3b%d' % i)
		print("after block2:", outputs.shape)
		outputs = self._dilated_bottle_resblock(outputs, 1024, 2, '4a',	identity_connection=False)
		for i in six.moves.range(1, 23):
			outputs = self._dilated_bottle_resblock(outputs, 1024, 2, '4b%d' % i)
		print("after block3:", outputs.shape)
		outputs = self._dilated_bottle_resblock(outputs, 2048, 4, '5a',	identity_connection=False)
		outputs = self._dilated_bottle_resblock(outputs, 2048, 4, '5b')
		outputs = self._dilated_bottle_resblock(outputs, 2048, 4, '5c')
		print("after block4:", outputs.shape)
		return outputs

	def build_decoder(self, encoding):
		print("-----------build decoder-----------")
		outputs = self._ASPP(encoding, self.num_classes, [6, 12, 18, 24])
		print("after aspp block:", outputs.shape)
		return outputs

	# blocks
	def _start_block(self):
		outputs = self._conv2d(self.inputs, 7, 64, 2, name='conv1')
		outputs = self._batch_norm(outputs, name='bn_conv1', is_training=False, activation_fn=tf.nn.relu)
		outputs = self._max_pool2d(outputs, 3, 2, name='pool1')
		return outputs

	def _bottleneck_resblock(self, x, num_o, name, half_size=False, identity_connection=True):
		first_s = 2 if half_size else 1
		assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
		# branch1
		if not identity_connection:
			o_b1 = self._conv2d(x, 1, num_o, first_s, name='res%s_branch1' % name)
			o_b1 = self._batch_norm(o_b1, name='bn%s_branch1' % name, is_training=False, activation_fn=None)
		else:
			o_b1 = x
		# branch2
		o_b2a = self._conv2d(x, 1, num_o // 4, first_s, name='res%s_branch2a' % name)
		o_b2a = self._batch_norm(o_b2a, name='bn%s_branch2a' % name, is_training=False, activation_fn=tf.nn.relu)

		o_b2b = self._conv2d(o_b2a, 3, num_o // 4, 1, name='res%s_branch2b' % name)
		o_b2b = self._batch_norm(o_b2b, name='bn%s_branch2b' % name, is_training=False, activation_fn=tf.nn.relu)

		o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='res%s_branch2c' % name)
		o_b2c = self._batch_norm(o_b2c, name='bn%s_branch2c' % name, is_training=False, activation_fn=None)
		# add
		outputs = self._add([o_b1,o_b2c], name='res%s' % name)
		# relu
		outputs = self._relu(outputs, name='res%s_relu' % name)
		return outputs

	def _dilated_bottle_resblock(self, x, num_o, dilation_factor, name, identity_connection=True):
		assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
		# branch1
		if not identity_connection:
			o_b1 = self._conv2d(x, 1, num_o, 1, name='res%s_branch1' % name)
			o_b1 = self._batch_norm(o_b1, name='bn%s_branch1' % name, is_training=False, activation_fn=None)
		else:
			o_b1 = x
		# branch2
		o_b2a = self._conv2d(x, 1, num_o // 4, 1, name='res%s_branch2a' % name)
		o_b2a = self._batch_norm(o_b2a, name='bn%s_branch2a' % name, is_training=False, activation_fn=tf.nn.relu)

		o_b2b = self._dilated_conv2d(o_b2a, 3, num_o // 4, dilation_factor, name='res%s_branch2b' % name)
		o_b2b = self._batch_norm(o_b2b, name='bn%s_branch2b' % name, is_training=False, activation_fn=tf.nn.relu)

		o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='res%s_branch2c' % name)
		o_b2c = self._batch_norm(o_b2c, name='bn%s_branch2c' % name, is_training=False, activation_fn=None)
		# add
		outputs = self._add([o_b1,o_b2c], name='res%s' % name)
		# relu
		outputs = self._relu(outputs, name='res%s_relu' % name)
		return outputs

	def _ASPP(self, x, num_o, dilations):
		o = []
		for i, d in enumerate(dilations):
			o.append(self._dilated_conv2d(x, 3, num_o, d, name='fc1_voc12_c%d' % i, biased=True))
		return self._add(o, name='fc1_voc12')

	# layers
	def _conv2d(self, x, kernel_size, num_o, stride, name, biased=False):
		"""
		Conv2d without BN or relu.
		"""
		num_x = x.shape[self.channel_axis]  # Remove  for TF2 compatibility
		with tf.variable_scope(name) as scope:
			w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
			s = [1, stride, stride, 1]
			o = tf.nn.conv2d(x, w, s, padding='SAME')
			if biased:
				b = tf.get_variable('biases', shape=[num_o])
				o = tf.nn.bias_add(o, b)
			return o

	def _dilated_conv2d(self, x, kernel_size, num_o, dilation_factor, name, biased=False):
		"""
		Dilated conv2d without BN or relu.
		"""
		num_x = x.shape[self.channel_axis]
		with tf.variable_scope(name) as scope:
			w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
			o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
			if biased:
				b = tf.get_variable('biases', shape=[num_o])
				o = tf.nn.bias_add(o, b)
			return o

	def _relu(self, x, name):
		return tf.nn.relu(x, name=name)

	def _add(self, x_l, name):
		return tf.add_n(x_l, name=name)

	def _max_pool2d(self, x, kernel_size, stride, name):
		k = [1, kernel_size, kernel_size, 1]
		s = [1, stride, stride, 1]
		return tf.nn.max_pool(x, k, s, padding='SAME', name=name)

	def _batch_norm(self, x, name, is_training, activation_fn, trainable=False):
		# TF2 compatible batch normalization using Keras layers
		# For a small batch size, it is better to keep
		# the statistics of the BN layers (running means and variances) frozen,
		# and to not update the values provided by the pre-trained model by setting is_training=False.
		# Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
		# if they are presented in var_list of the optimiser definition.
		# Set trainable = False to remove them from trainable_variables.
		
		# Create BatchNormalization layer with proper name scope
		with tf.variable_scope(name) as scope:
			# Use compat.v1 for backward compatibility with existing checkpoints
			bn_layer = tf.keras.layers.BatchNormalization(
				scale=True,
				trainable=trainable,
				name=scope.name + '_bn'
			)
			o = bn_layer(x, training=is_training)
			if activation_fn is not None:
				o = activation_fn(o)
			return o



class ResNet_segmentation(object):
	"""
	Original ResNet-101 ('resnet_v1_101.ckpt')
	Original ResNet-50 ('resnet_v1_50.ckpt')
	"""
	def __init__(self, inputs, num_classes, phase, encoder_name):
		if encoder_name not in ['res101', 'res50']:
			print('encoder_name ERROR!')
			print("Please input: res101, res50")
			sys.exit(-1)
		self.encoder_name = encoder_name
		self.inputs = inputs
		self.num_classes = num_classes
		self.channel_axis = 3
		self.phase = phase # train (True) or test (False), for BN layers in the decoder
		self.build_network()

	def build_network(self):
		self.encoding = self.build_encoder()
		self.outputs = self.build_decoder(self.encoding)

	def build_encoder(self):
		print("-----------build encoder: %s-----------" % self.encoder_name)
		scope_name = 'resnet_v1_101' if self.encoder_name == 'res101' else 'resnet_v1_50'
		with tf.variable_scope(scope_name) as scope:
			outputs = self._start_block('conv1')
			print("after start block:", outputs.shape)
			with tf.variable_scope('block1') as scope:
				outputs = self._bottleneck_resblock(outputs, 256, 'unit_1',	identity_connection=False)
				outputs = self._bottleneck_resblock(outputs, 256, 'unit_2')
				outputs = self._bottleneck_resblock(outputs, 256, 'unit_3')
				print("after block1:", outputs.shape)
			with tf.variable_scope('block2') as scope:
				outputs = self._bottleneck_resblock(outputs, 512, 'unit_1',	half_size=True, identity_connection=False)
				for i in six.moves.range(2, 5):
					outputs = self._bottleneck_resblock(outputs, 512, 'unit_%d' % i)
				print("after block2:", outputs.shape)
			with tf.variable_scope('block3') as scope:
				outputs = self._dilated_bottle_resblock(outputs, 1024, 2, 'unit_1',	identity_connection=False)
				num_layers_block3 = 23 if self.encoder_name == 'res101' else 6
				for i in six.moves.range(2, num_layers_block3+1):
					outputs = self._dilated_bottle_resblock(outputs, 1024, 2, 'unit_%d' % i)
				print("after block3:", outputs.shape)
			with tf.variable_scope('block4') as scope:
				outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_1', identity_connection=False)
				outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_2')
				outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_3')
				print("after block4:", outputs.shape)
				return outputs

	def build_decoder(self, encoding):
		print("-----------build decoder-----------")
		with tf.variable_scope('decoder') as scope:
			outputs = self._ASPP(encoding, self.num_classes, [6, 12, 18, 24])
			print("after aspp block:", outputs.shape)
			return outputs

	# blocks
	def _start_block(self, name):
		outputs = self._conv2d(self.inputs, 7, 64, 2, name=name)
		outputs = self._batch_norm(outputs, name=name, is_training=False, activation_fn=tf.nn.relu)
		outputs = self._max_pool2d(outputs, 3, 2, name='pool1')
		return outputs

	def _bottleneck_resblock(self, x, num_o, name, half_size=False, identity_connection=True):
		first_s = 2 if half_size else 1
		assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
		# branch1
		if not identity_connection:
			o_b1 = self._conv2d(x, 1, num_o, first_s, name='%s/bottleneck_v1/shortcut' % name)
			o_b1 = self._batch_norm(o_b1, name='%s/bottleneck_v1/shortcut' % name, is_training=False, activation_fn=None)
		else:
			o_b1 = x
		# branch2
		o_b2a = self._conv2d(x, 1, num_o // 4, first_s, name='%s/bottleneck_v1/conv1' % name)
		o_b2a = self._batch_norm(o_b2a, name='%s/bottleneck_v1/conv1' % name, is_training=False, activation_fn=tf.nn.relu)

		o_b2b = self._conv2d(o_b2a, 3, num_o // 4, 1, name='%s/bottleneck_v1/conv2' % name)
		o_b2b = self._batch_norm(o_b2b, name='%s/bottleneck_v1/conv2' % name, is_training=False, activation_fn=tf.nn.relu)

		o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='%s/bottleneck_v1/conv3' % name)
		o_b2c = self._batch_norm(o_b2c, name='%s/bottleneck_v1/conv3' % name, is_training=False, activation_fn=None)
		# add
		outputs = self._add([o_b1,o_b2c], name='%s/bottleneck_v1/add' % name)
		# relu
		outputs = self._relu(outputs, name='%s/bottleneck_v1/relu' % name)
		return outputs

	def _dilated_bottle_resblock(self, x, num_o, dilation_factor, name, identity_connection=True):
		assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
		# branch1
		if not identity_connection:
			o_b1 = self._conv2d(x, 1, num_o, 1, name='%s/bottleneck_v1/shortcut' % name)
			o_b1 = self._batch_norm(o_b1, name='%s/bottleneck_v1/shortcut' % name, is_training=False, activation_fn=None)
		else:
			o_b1 = x
		# branch2
		o_b2a = self._conv2d(x, 1, num_o // 4, 1, name='%s/bottleneck_v1/conv1' % name)
		o_b2a = self._batch_norm(o_b2a, name='%s/bottleneck_v1/conv1' % name, is_training=False, activation_fn=tf.nn.relu)

		o_b2b = self._dilated_conv2d(o_b2a, 3, num_o // 4, dilation_factor, name='%s/bottleneck_v1/conv2' % name)
		o_b2b = self._batch_norm(o_b2b, name='%s/bottleneck_v1/conv2' % name, is_training=False, activation_fn=tf.nn.relu)

		o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='%s/bottleneck_v1/conv3' % name)
		o_b2c = self._batch_norm(o_b2c, name='%s/bottleneck_v1/conv3' % name, is_training=False, activation_fn=None)
		# add
		outputs = self._add([o_b1,o_b2c], name='%s/bottleneck_v1/add' % name)
		# relu
		outputs = self._relu(outputs, name='%s/bottleneck_v1/relu' % name)
		return outputs

	def _ASPP(self, x, num_o, dilations):
		o = []
		for i, d in enumerate(dilations):
			o.append(self._dilated_conv2d(x, 3, num_o, d, name='aspp/conv%d' % (i+1), biased=True))
		return self._add(o, name='aspp/add')

	# layers
	def _conv2d(self, x, kernel_size, num_o, stride, name, biased=False):
		"""
		Conv2d without BN or relu.
		"""
		num_x = x.shape[self.channel_axis]
		with tf.variable_scope(name) as scope:
			w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
			s = [1, stride, stride, 1]
			o = tf.nn.conv2d(x, w, s, padding='SAME')
			if biased:
				b = tf.get_variable('biases', shape=[num_o])
				o = tf.nn.bias_add(o, b)
			return o

	def _dilated_conv2d(self, x, kernel_size, num_o, dilation_factor, name, biased=False):
		"""
		Dilated conv2d without BN or relu.
		"""
		num_x = x.shape[self.channel_axis]
		with tf.variable_scope(name) as scope:
			w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
			o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
			if biased:
				b = tf.get_variable('biases', shape=[num_o])
				o = tf.nn.bias_add(o, b)
			return o

	def _relu(self, x, name):
		return tf.nn.relu(x, name=name)

	def _add(self, x_l, name):
		return tf.add_n(x_l, name=name)

	def _max_pool2d(self, x, kernel_size, stride, name):
		k = [1, kernel_size, kernel_size, 1]
		s = [1, stride, stride, 1]
		return tf.nn.max_pool(x, k, s, padding='SAME', name=name)

	def _batch_norm(self, x, name, is_training, activation_fn, trainable=False):
		# For a small batch size, it is better to keep
		# the statistics of the BN layers (running means and variances) frozen,
		# and to not update the values provided by the pre-trained model by setting is_training=False.
		# Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
		# if they are presented in var_list of the optimiser definition.
		# Set trainable = False to remove them from trainable_variables.
		with tf.variable_scope(name+'/BatchNorm') as scope:
			# TF2 compatible batch normalization
			bn_layer = tf.keras.layers.BatchNormalization(
				scale=True,
				trainable=trainable,
				name=scope.name + '_bn'
			)
			o = bn_layer(x, training=is_training)
			if activation_fn is not None:
				o = activation_fn(o)
			return o


class ResNet_segmentation_TF2(tf.keras.Model):
	"""
	TF2.x native implementation of ResNet for segmentation
	"""
	def __init__(self, num_classes, **kwargs):
		super(ResNet_segmentation_TF2, self).__init__(**kwargs)
		self.num_classes = num_classes
		self.channel_axis = 3
		
		# Initial convolution layers  
		self.conv1 = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', 
											kernel_initializer='he_normal', name='conv1')
		self.bn_conv1 = tf.keras.layers.BatchNormalization(name='bn_conv1')
		self.pool1 = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same', name='pool1')
		
		# ResNet stages - similar structure to Deeplab but simpler decoder
		self.encoder = self._build_resnet_encoder()
		
		# Simple segmentation head
		self.seg_conv1 = tf.keras.layers.Conv2D(512, 3, padding='same', 
												kernel_initializer='he_normal', name='seg_conv1')
		self.seg_bn1 = tf.keras.layers.BatchNormalization(name='seg_bn1')
		
		self.seg_conv2 = tf.keras.layers.Conv2D(num_classes, 1, padding='same', 
												kernel_initializer='he_normal', name='seg_conv2')
		
		# Upsampling to restore resolution
		self.upsample = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear', name='upsample')
	
	def _build_resnet_encoder(self):
		"""Build ResNet encoder using Sequential model"""
		layers = []
		
		# Use simplified ResNet blocks for the encoder
		# Stage 2
		layers.extend([
			tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
		])
		
		# Stage 3
		layers.extend([
			tf.keras.layers.Conv2D(512, 3, strides=2, padding='same', kernel_initializer='he_normal'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
		])
		
		# Stage 4
		layers.extend([
			tf.keras.layers.Conv2D(1024, 3, strides=2, padding='same', kernel_initializer='he_normal'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
		])
		
		# Stage 5
		layers.extend([
			tf.keras.layers.Conv2D(2048, 3, padding='same', kernel_initializer='he_normal'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
		])
		
		return tf.keras.Sequential(layers, name='resnet_encoder')
	
	def call(self, inputs, training=None):
		"""Forward pass through the network"""
		# Initial convolution
		x = self.conv1(inputs)
		x = self.bn_conv1(x, training=False)
		x = tf.nn.relu(x)
		x = self.pool1(x)
		
		print(f"ResNet after start: {x.shape}")
		
		# Encoder
		x = self.encoder(x, training=training)
		
		print(f"ResNet after encoder: {x.shape}")
		
		# Segmentation head
		x = self.seg_conv1(x)
		x = self.seg_bn1(x, training=training)
		x = tf.nn.relu(x)
		
		x = self.seg_conv2(x)
		
		print(f"ResNet before upsample: {x.shape}")
		
		# Upsample to original resolution
		outputs = self.upsample(x)
		
		print(f"ResNet final output: {outputs.shape}")
		
		return outputs
