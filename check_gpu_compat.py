import tensorflow as tf
if __name__ == "__main__":
    print("TF Version:", tf.__version__)
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))