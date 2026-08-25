from datetime import datetime
import os
import sys
import time
import numpy as np
import tensorflow as tf
from PIL import Image

try:
	from tqdm import tqdm as _tqdm
	HAS_TQDM = True
except ImportError:
	HAS_TQDM = False

# Handle both relative and absolute imports
try:
	# Relative imports when used as a package
	from .models import DeepLabV2, ResNetSegmentation
	from .utils.image_reader import ImageReader
	from .utils import (
		prepare_label, decode_labels, read_labeled_image_list, 
		create_dataset, write_log
	)
except ImportError:
    # Fallback to absolute imports when called as a script
	from models import DeepLabV2, ResNetSegmentation
	from utils.image_reader import ImageReader, IMG_MEAN
	from utils import (
		prepare_label, decode_labels, read_labeled_image_list,
		create_dataset, write_log
	)




"""
This script trains or evaluates the model on augmented PASCAL VOC 2012 dataset.
The training set contains 10581 training images.
The validation set contains 1449 validation images.

Training:
'poly' learning rate
different learning rates for different layers
"""



IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

class Model(object):

	def __init__(self, conf):
		self.conf = conf
		# TF2.x uses eager execution by default, no session needed
		# Initialize metrics for TF2
		self.accuracy_metric = tf.keras.metrics.Accuracy()
		self.miou_metric = tf.keras.metrics.MeanIoU(num_classes=conf.num_classes)
		
		# Initialize summary writer for TF2
		self.summary_writer = tf.summary.create_file_writer(self.conf.logdir)
		
		# Create model components
		self.model = None
		self.optimizer_encoder = None
		self.optimizer_decoder_w = None 
		self.optimizer_decoder_b = None
	
	def build_model(self, input_shape):
		"""Build the DeepLab model architecture for TF2."""
		# Create network
		if self.conf.encoder_name not in ['res101', 'res50', 'deeplab']:
			print('encoder_name ERROR!')
			print("Please input: res101, res50, or deeplab")
			sys.exit(-1)
		elif self.conf.encoder_name == 'deeplab':
			self.model = DeepLabV2(num_classes=self.conf.num_classes)
		else:
			self.model = ResNetSegmentation(
				num_classes=self.conf.num_classes, 
				encoder_name=self.conf.encoder_name,
				name='resnet_v1_50' if self.conf.encoder_name == 'res50' else 'resnet_v1_101'
			)
		
		return self.model
	
	def setup_optimizers(self):
		# Create optimizers for different learning rates
		base_lr = self.conf.learning_rate
		
		self.optimizer_encoder = tf.keras.optimizers.SGD(
			learning_rate=base_lr, 
			momentum=self.conf.momentum
		)
		self.optimizer_decoder_w = tf.keras.optimizers.SGD(
			learning_rate=base_lr * 10.0,
			momentum=self.conf.momentum
		)
		self.optimizer_decoder_b = tf.keras.optimizers.SGD(
			learning_rate=base_lr * 20.0,
			momentum=self.conf.momentum
		)

	# train
	def train(self):		
		# Setup model and data pipeline
		self.train_setup_tf2()
		
		# Train!
		for step in range(self.conf.num_steps + 1):
			start_time = time.time()
			
			# Get batch from dataset
			try:
				batch = next(self.train_dataset_iter)
			except StopIteration:
				# Reset iterator if dataset is exhausted
				self.train_dataset_iter = iter(self.train_dataset)
				batch = next(self.train_dataset_iter)
			
			# Training step
			loss_value = self.train_step(batch, step)
			
			# Logging and saving
			if step % self.conf.save_interval == 0:
				self.log_training_step(batch, step, loss_value)
				self.save_checkpoint(step)
			
			duration = time.time() - start_time
			print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
			write_log('{:d}, {:.3f}'.format(step, loss_value), self.conf.logfile)
	
	@tf.function
	def train_step(self, batch, step):
		"""Single training step using tf.function for performance."""
		images, labels = batch
		
		# Calculate current learning rate with polynomial decay
		current_lr = self.conf.learning_rate * tf.pow(
			(1 - tf.cast(step, tf.float32) / self.conf.num_steps), 
			self.conf.power
		)
		
		# Update optimizer learning rates
		self.optimizer_encoder.learning_rate = current_lr
		self.optimizer_decoder_w.learning_rate = current_lr * 10.0
		self.optimizer_decoder_b.learning_rate = current_lr * 20.0
		
		with tf.GradientTape() as tape:
			# Forward pass
			predictions = self.model(images, training=True)
			
			# Compute loss
			loss = self.compute_loss(predictions, labels)
		
		# Compute gradients
		gradients = tape.gradient(loss, self.model.trainable_variables)
		
		# Apply gradients with different optimizers for different layers
		# This is a simplified version - will need refinement based on actual network structure
		self.optimizer_encoder.apply_gradients(zip(gradients, self.model.trainable_variables))
		
		return loss
	
	def compute_loss(self, predictions, labels):
		"""Compute segmentation loss."""
		# Prepare labels
		output_shape = tf.shape(predictions)
		output_size = (output_shape[1], output_shape[2])
		
		label_proc = prepare_label(labels, output_size, num_classes=self.conf.num_classes, one_hot=False)
		raw_gt = tf.reshape(label_proc, [-1])
		indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, self.conf.num_classes - 1)), 1)
		gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
		
		raw_prediction = tf.reshape(predictions, [-1, self.conf.num_classes])
		prediction = tf.gather(raw_prediction, indices)
		
		# Pixel-wise softmax cross entropy loss
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
		
		# L2 regularization
		l2_losses = [self.conf.weight_decay * tf.nn.l2_loss(v) 
					for v in self.model.trainable_variables 
					if 'kernel' in v.name or 'weight' in v.name]
		
		# Total loss
		total_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses) if l2_losses else tf.reduce_mean(loss)
		
		return total_loss
	
	def log_training_step(self, batch, step, loss_value):
		"""Log training metrics and images."""
		images, labels = batch
		
		# Get predictions for visualization
		predictions = self.model(images, training=False)
		
		# Write summaries
		with self.summary_writer.as_default():
			tf.summary.scalar('loss', loss_value, step=step)
			
			# Image summaries (simplified - may need adjustment based on actual preprocessing)
			if step % (self.conf.save_interval * 5) == 0:  # Less frequent image logging
				# Convert predictions to visualization format
				pred_vis = tf.argmax(predictions, axis=-1)
				pred_vis = tf.expand_dims(pred_vis, axis=-1)
				
				tf.summary.image('images', images[:2], step=step, max_outputs=2)
				tf.summary.image('labels', labels[:2], step=step, max_outputs=2)
				tf.summary.image('predictions', tf.cast(pred_vis[:2], tf.uint8), step=step, max_outputs=2)
			
			self.summary_writer.flush()
	
	def save_checkpoint(self, step):
		"""Save model checkpoint."""
		if not os.path.exists(self.conf.modeldir):
			os.makedirs(self.conf.modeldir)
		
		checkpoint_path = os.path.join(self.conf.modeldir, f'model_step_{step}')
		
		# Save using tf.train.Checkpoint
		checkpoint = tf.train.Checkpoint(
			model=self.model,
			optimizer_encoder=self.optimizer_encoder,
			optimizer_decoder_w=self.optimizer_decoder_w,
			optimizer_decoder_b=self.optimizer_decoder_b
		)
		checkpoint.save(checkpoint_path)
		print(f'Checkpoint saved at step {step}')
	
	def load_checkpoint(self, checkpoint_path):
		"""Load model checkpoint for TF2."""
		import os
		
		# Ensure model is built
		if self.model is None:
			# Build model with default input shape - will be resized during inference
			self.build_model(input_shape=(None, None, 3))
		
		# Ensure optimizers are initialized
		if not hasattr(self, 'optimizer_encoder') or self.optimizer_encoder is None:
			self.setup_optimizers()
		
		# Check if TF2 checkpoint exists
		if os.path.exists(checkpoint_path + '.index'):
			
			checkpoint = tf.train.Checkpoint(
				model=self.model,
				optimizer_encoder=self.optimizer_encoder,
				optimizer_decoder_w=self.optimizer_decoder_w,
				optimizer_decoder_b=self.optimizer_decoder_b
			)
			
			# Use expect_partial() to suppress warnings for missing/extra variables
			print(f"Loading checkpoint from {checkpoint_path}")
			status = checkpoint.restore(checkpoint_path)
			print(f"Restored TF2 model from {checkpoint_path}")
			return
		else:
			raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

	# evaluate
	def test(self):
		# Setup model and data pipeline for testing
		self.test_setup_tf2()
		
		# Load checkpoint
		checkpoint_path = os.path.join(self.conf.modeldir, f'model_step_{self.conf.valid_step}')
		self.load_checkpoint(checkpoint_path)
		
		# Reset metrics
		self.accuracy_metric.reset_states()
		self.miou_metric.reset_states()
		
		# Test!
		confusion_matrix = np.zeros((self.conf.num_classes, self.conf.num_classes), dtype=np.int)
		
		for step in range(self.conf.valid_num_steps):
			try:
				batch = next(self.test_dataset_iter)
			except StopIteration:
				break
			
			# Get predictions
			images, labels = batch
			predictions = self.model(images, training=False)
			
			# Convert to class predictions
			pred_classes = tf.argmax(predictions, axis=-1)
			
			# Flatten for metrics
			pred_flat = tf.reshape(pred_classes, [-1])
			labels_flat = tf.reshape(labels, [-1])
			
			# Create mask for valid labels
			valid_mask = tf.less_equal(labels_flat, self.conf.num_classes - 1)
			valid_labels = tf.boolean_mask(labels_flat, valid_mask)
			valid_preds = tf.boolean_mask(pred_flat, valid_mask)
			
			# Update metrics
			self.accuracy_metric.update_state(valid_labels, valid_preds)
			self.miou_metric.update_state(valid_labels, valid_preds)
			
			# Update confusion matrix
			c_matrix = tf.math.confusion_matrix(
				valid_labels, valid_preds, 
				num_classes=self.conf.num_classes
			)
			confusion_matrix += c_matrix.numpy()
			
			if step % 100 == 0:
				print('step {:d}'.format(step))
		
		# Print results
		print('Pixel Accuracy: {:.3f}'.format(self.accuracy_metric.result().numpy()))
		print('Mean IoU: {:.3f}'.format(self.miou_metric.result().numpy()))
		self.compute_IoU_per_class(confusion_matrix)

	# prediction
	def predict(self):
		self._predict_impl()

	def _predict_impl(self):
		# Setup model and data pipeline for prediction
		self.prediction_setup()

		# Load checkpoint
		checkpoint_path = os.path.join(self.conf.modeldir, 'deeplabv2')
		self.load_checkpoint(checkpoint_path)

		# Get image name list
		image_list, _ = read_labeled_image_list('', self.conf.test_data_list)

		# Create output directories
		if not os.path.exists(self.conf.out_dir):
			os.makedirs(self.conf.out_dir)
			os.makedirs(self.conf.out_dir + '/prediction')
			if self.conf.visual:
				os.makedirs(self.conf.out_dir + '/visual_prediction')

		# Setup Girder progress reporting if credentials provided
		girder_gc = None
		girder_job_id = getattr(self.conf, 'girder_job_id', None)
		if girder_job_id and getattr(self.conf, 'girder_api_url', None) and getattr(self.conf, 'girder_token', None):
			try:
				from girder_client import GirderClient
				girder_gc = GirderClient(apiUrl=self.conf.girder_api_url)
				girder_gc.setToken(self.conf.girder_token)
			except Exception:
				girder_gc = None

		total_images = len(image_list)
		total_steps = min(self.conf.test_num_steps, -(-total_images // self.conf.batch_size))

		pbar = _tqdm(
			total=total_images,
			desc='Total WSI progress',
			unit='image',
			colour='green',
			file=sys.stdout,
			dynamic_ncols=False,
			ncols=80,
		) if HAS_TQDM else None

		# Predict!
		for step in range(self.conf.test_num_steps):
			try:
					batch = next(self.predict_dataset_iter)
			except StopIteration:
					break

			images, _ = batch
			batch_size = images.shape[0]

			# Get predictions
			predictions = self.model(images, training=False)
			pred_classes = tf.argmax(predictions, axis=-1)

			# Process each image in the batch
			imgs_this_batch = 0
			for i in range(batch_size):
				# Convert to numpy for saving
				pred_np = pred_classes.numpy()[i]  # Get i-th image in batch

				# Calculate the actual image index
				img_index = step * self.conf.batch_size + i
				if img_index >= len(image_list):
						break  # Don't process beyond available images

				# Get image name
				img_name = image_list[img_index].split('/')[2].split('.')[0] if '/' in image_list[img_index] else image_list[img_index].split('.')[0]

				# Save raw predictions
				im = Image.fromarray(pred_np.astype(np.uint8), mode='L')
				filename = f'/{img_name}_mask.png'
				im.save(self.conf.out_dir + '/prediction' + filename)

				# Save predictions for visualization
				if self.conf.visual:
					msk = decode_labels(tf.expand_dims(tf.expand_dims(pred_classes[i:i+1], axis=-1), axis=0),
														num_classes=self.conf.num_classes)
					im = Image.fromarray(msk[0], mode='RGB')
					filename = f'/{img_name}_mask_visual.png'
					im.save(self.conf.out_dir + '/visual_prediction' + filename)

				imgs_this_batch += 1
				if pbar is not None:
					pbar.update(1)

			if pbar is None:
				imgs_done = min((step + 1) * self.conf.batch_size, total_images)
				pct = int(imgs_done / total_images * 100) if total_images > 0 else 0
				print(f'Step {step + 1}/{total_steps} — {imgs_done}/{total_images} images ({pct}%)')
				sys.stdout.flush()

			if girder_gc and girder_job_id:
				try:
					imgs_done = min((step + 1) * self.conf.batch_size, total_images)
					pct = int(imgs_done / total_images * 100) if total_images > 0 else 0
					girder_gc.patch(f'/job/{girder_job_id}', data={
						'progress': {'current': imgs_done, 'total': total_images, 'message': f'Predicting: {pct}%'}
					})
				except Exception:
					pass

		if pbar is not None:
			pbar.close()

		print('The output files have been saved to {}'.format(self.conf.out_dir))

	def create_dataset(self, data_dir, data_list, input_size=None, is_training=False):
		# Use the new utility function for creating TF2 datasets
		dataset = create_dataset(
			data_dir=data_dir,
			data_list=data_list,
			input_size=input_size,
			batch_size=self.conf.batch_size,
			random_scale=self.conf.random_scale if is_training else False,
			random_mirror=self.conf.random_mirror if is_training else False,
			ignore_label=self.conf.ignore_label,
			img_mean=tf.constant(IMG_MEAN, dtype=tf.float32),
			shuffle=is_training,
			repeat=is_training
		)
		
		return dataset

	def train_setup_tf2(self):
		"""Setup training pipeline for TF2."""
		tf.random.set_seed(self.conf.random_seed)
		
		# Create directories
		if not os.path.exists(self.conf.logdir):
			os.makedirs(self.conf.logdir)
		
		# Input size
		input_size = (self.conf.input_height, self.conf.input_width)
		
		# Create training dataset
		self.train_dataset = self.create_dataset(
			self.conf.data_dir,
			self.conf.data_list,
			input_size=input_size,
			is_training=True
		)
		self.train_dataset_iter = iter(self.train_dataset)
		
		# Build model
		self.build_model(input_shape=(self.conf.input_height, self.conf.input_width, 3))
		
		# Setup optimizers
		self.setup_optimizers()
	
	def test_setup_tf2(self):
		"""Setup testing pipeline for TF2."""
		# Create validation dataset
		self.test_dataset = self.create_dataset(
			self.conf.data_dir,
			self.conf.valid_data_list,
			input_size=None,  # Variable size for testing
			is_training=False
		)
		self.test_dataset_iter = iter(self.test_dataset)
		
		# Build model if not already built
		if self.model is None:
			self.build_model(input_shape=(None, None, 3))
		
		# Setup optimizers (needed for checkpoint loading)
		self.setup_optimizers()
	
	def prediction_setup(self):
		"""Setup prediction pipeline for TF2."""
		# Create prediction dataset
		self.predict_dataset = self.create_dataset(
			self.conf.data_dir,
			self.conf.test_data_list,
			input_size=None,  # Variable size for prediction
			is_training=False
		)
		self.predict_dataset_iter = iter(self.predict_dataset)

		# Build model if not already built
		if self.model is None:
			self.build_model(input_shape=(None, None, 3))

		# Setup optimizers (needed for checkpoint loading)
		self.setup_optimizers()

	def compute_IoU_per_class(self, confusion_matrix):
		"""Compute IoU per class from confusion matrix."""
		mIoU = 0
		for i in range(self.conf.num_classes):
			# IoU = true_positive / (true_positive + false_positive + false_negative)
			TP = confusion_matrix[i,i]
			FP = np.sum(confusion_matrix[:, i]) - TP
			FN = np.sum(confusion_matrix[i]) - TP
			IoU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
			print ('class %d: %.3f' % (i, IoU))
			mIoU += IoU / self.conf.num_classes
		print ('mIoU: %.3f' % mIoU)
		return mIoU
