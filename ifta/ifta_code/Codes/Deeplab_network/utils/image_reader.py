import os

import numpy as np
import tensorflow as tf

def image_scaling(img, label):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """
    
    scale = tf.random.uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
    h_new = tf.cast(tf.multiply(tf.cast(tf.shape(img)[0], tf.float32), scale), tf.int32)
    w_new = tf.cast(tf.multiply(tf.cast(tf.shape(img)[1], tf.float32), scale), tf.int32)
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), axis=[1])
    img = tf.image.resize(img, new_shape)
    label = tf.image.resize(tf.expand_dims(label, 0), new_shape, method='nearest')
    label = tf.squeeze(label, axis=[0])
   
    return img, label

def image_mirroring(img, label):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """
    
    distort_left_right_random = tf.random.uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    return img, label

def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat([image, label], axis=2) 
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))
    
    last_image_dim = tf.shape(image)[-1]
    # last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.image.random_crop(combined_pad, [crop_h, crop_w, 4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)
    
    # Set static shape so that tensorflow knows shape at compile time. 
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h,crop_w, 1))
    return img_crop, label_crop  

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line.strip("\n").rsplit(' ', 1)
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
    return images, masks

def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, ignore_label, img_mean): # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.
      
    Returns:
      Two tensors: the decoded image and its mask.
    """

    img_contents = tf.io.read_file(input_queue[0])
    label_contents = tf.io.read_file(input_queue[1])
    
    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(img, 3, axis=2)
    img = tf.cast(tf.concat([img_b, img_g, img_r], axis=2), dtype=tf.float32)
    # Extract mean.
    img -= img_mean

    label = tf.image.decode_png(label_contents, channels=1)

    if input_size is not None:
        h, w = input_size

        # Randomly scale the images and labels.
        if random_scale:
            img, label = image_scaling(img, label)

        # Randomly mirror the images and labels.
        if random_mirror:
            img, label = image_mirroring(img, label)

        # Randomly crops the images and labels.
        img, label = random_crop_and_pad_image_and_labels(img, label, h, w, ignore_label)

    return img, label

IMG_MEAN = tf.constant([104.00698793, 116.66876762, 122.67891434], dtype=tf.float32)

class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk using TF2.x tf.data.Dataset patterns.
    '''

    def __init__(self, data_dir, data_list, input_size, 
                 random_scale, random_mirror, ignore_label, img_mean=IMG_MEAN, coord=None):
        '''Initialise an ImageReader.
        
        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          ignore_label: index of label to ignore during the training.
          img_mean: vector of mean colour values.
          coord: TensorFlow queue coordinator (deprecated in TF2, kept for compatibility).
        '''
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.random_scale = random_scale
        self.random_mirror = random_mirror
        self.ignore_label = ignore_label
        self.img_mean = img_mean
        self.coord = coord  # Keep for backward compatibility but not used in TF2
        
        self.image_list, self.label_list = read_labeled_image_list(self.data_dir, self.data_list)
        
        # Create tf.data.Dataset for TF2
        self.dataset = tf.data.Dataset.from_tensor_slices((self.image_list, self.label_list))
        
        # Shuffle if training
        if input_size is not None:
            self.dataset = self.dataset.shuffle(buffer_size=len(self.image_list))
            
        # Map the load and preprocess function
        self.dataset = self.dataset.map(
            self._load_and_preprocess_image, 
            num_parallel_calls=tf.data.AUTOTUNE
        )

    def _load_and_preprocess_image(self, image_path, label_path):
        """Load and preprocess a single image and label pair."""
        # Read files
        img_contents = tf.io.read_file(image_path)
        label_contents = tf.io.read_file(label_path)
        
        # Decode images
        img = tf.image.decode_jpeg(img_contents, channels=3)
        img_r, img_g, img_b = tf.split(img, 3, axis=2)
        img = tf.cast(tf.concat([img_b, img_g, img_r], axis=2), dtype=tf.float32)
        # Extract mean
        img -= self.img_mean

        label = tf.image.decode_png(label_contents, channels=1)

        if self.input_size is not None:
            h, w = self.input_size

            # Randomly scale the images and labels
            if self.random_scale:
                img, label = image_scaling(img, label)

            # Randomly mirror the images and labels
            if self.random_mirror:
                img, label = image_mirroring(img, label)

            # Randomly crops the images and labels
            img, label = random_crop_and_pad_image_and_labels(img, label, h, w, self.ignore_label)

        return img, label

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch using tf.data.Dataset.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        
        # Batch the dataset
        batched_dataset = self.dataset.batch(num_elements)
        
        # Get one batch (for compatibility with the old interface)
        # Note: In TF2, you'd typically iterate over the dataset instead
        for image_batch, label_batch in batched_dataset.take(1):
            return image_batch, label_batch
            
    def get_dataset(self):
        """Return the tf.data.Dataset for use with TF2 training loops."""
        return self.dataset


def create_dataset(data_dir, data_list, input_size=None, batch_size=1, 
                      random_scale=False, random_mirror=False, ignore_label=255, 
                      img_mean=None, shuffle=True, repeat=True):
    """
    Create a TF2.x tf.data.Dataset for training/testing.
    
    Args:
        data_dir: path to the directory with images and masks.
        data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
        input_size: a tuple with (height, width) values.
        batch_size: the batch size.
        random_scale: whether to randomly scale the images.
        random_mirror: whether to randomly mirror the images.
        ignore_label: index of label to ignore during training.
        img_mean: vector of mean colour values.
        shuffle: whether to shuffle the dataset.
        repeat: whether to repeat the dataset indefinitely.
    
    Returns:
        A batched tf.data.Dataset.
    """
    if img_mean is None:
        img_mean = tf.constant(IMG_MEAN, dtype=tf.float32)
    
    # Read the file list
    image_list, label_list = read_labeled_image_list(data_dir, data_list)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list))
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_list))
    
    # Repeat if requested
    if repeat:
        dataset = dataset.repeat()
    
    # Map the preprocessing function
    def preprocess_fn(image_path, label_path):
        # Read files
        img_contents = tf.io.read_file(image_path)
        label_contents = tf.io.read_file(label_path)
        
        # Decode images
        img = tf.image.decode_jpeg(img_contents, channels=3)
        img_r, img_g, img_b = tf.split(img, 3, axis=2)
        img = tf.cast(tf.concat([img_b, img_g, img_r], axis=2), dtype=tf.float32)
        # Extract mean
        img -= img_mean

        label = tf.image.decode_png(label_contents, channels=1)

        if input_size is not None:
            h, w = input_size

            # Randomly scale the images and labels
            if random_scale:
                img, label = image_scaling(img, label)

            # Randomly mirror the images and labels
            if random_mirror:
                img, label = image_mirroring(img, label)

            # Randomly crops the images and labels
            img, label = random_crop_and_pad_image_and_labels(img, label, h, w, ignore_label)

        return img, label
    
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
