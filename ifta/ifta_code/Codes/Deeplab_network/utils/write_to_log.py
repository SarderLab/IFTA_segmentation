import os
import datetime
import tensorflow as tf

def write_log(str, filename):
    """Write a string to a log file with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {str}"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'a') as f:
        f.write(log_entry + "\n")

def write_tf_summary(summary_writer, tag, value, step):
    """Write a scalar summary using TF2.x summary writer."""
    with summary_writer.as_default():
        tf.summary.scalar(tag, value, step=step)
        summary_writer.flush()

def write_image_summary(summary_writer, tag, image, step, max_outputs=3):
    """Write an image summary using TF2.x summary writer."""
    with summary_writer.as_default():
        tf.summary.image(tag, image, step=step, max_outputs=max_outputs)
        summary_writer.flush()

def write_histogram_summary(summary_writer, tag, values, step):
    """Write a histogram summary using TF2.x summary writer."""
    with summary_writer.as_default():
        tf.summary.histogram(tag, values, step=step)
        summary_writer.flush()
