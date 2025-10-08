import argparse
import os
import sys
import tensorflow as tf

# Add current directory to path to enable proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model import Model

def configure(args):
    """Convert argparse args to a configuration object for backward compatibility."""
    class Config:
        def __init__(self, args):
            # Copy all args attributes to config
            for key, value in vars(args).items():
                setattr(self, key, value)
    
    return Config(args)

def setup_gpu(gpu_id):
    """Setup GPU with memory growth for TF2."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Configure GPU memory growth to prevent allocation of all GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU {gpu_id} configured with memory growth enabled")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPUs found, using CPU")


def validate_config(args):
    """Validate configuration parameters.""" 
    if args.option not in ['train', 'test', 'predict']:
        raise ValueError(f"Invalid option: {args.option}. Must be 'train', 'test', or 'predict'")
    
    if not os.path.exists(args.data_dir):
        print(f"Warning: Data directory does not exist: {args.data_dir}")
    
    if args.option in ['test', 'predict'] and not os.path.exists(args.test_data_list):
        print(f"Warning: Test data list does not exist: {args.test_data_list}")


def main(args):
    try:
        # Setup GPU configuration
        setup_gpu(args.gpu)
        
        # Validate configuration
        validate_config(args)
        
        # Create model with TF2-compatible configuration (no session needed)
        config = configure(args)
        model = Model(config)  # Remove sess parameter
        
        # Execute the requested operation
        getattr(model, args.option)()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--option', dest='option', type=str, default='train',
		help='actions: train, test, or predict')
	parser.add_argument('--out_dir', dest='out_dir', type=str, default='output',
		help='directory for saving testing outputs')
	parser.add_argument('--test_step', dest='test_step', type=int, default=00000,
		help='checkpoint number for testing/validation')
	parser.add_argument('--test_num_steps', dest='test_num_steps', type=int, default=81605,
		help='number of testing/validation samples')
	parser.add_argument('--test_data_list', dest='test_data_list', type=str, default='./dataset/test.txt',
		help='testing/validation data list filename')
	parser.add_argument('--visual', dest='visual', type=bool, default=False,
		help='whether to save predictions for visualization')
	parser.add_argument('--modeldir', dest='modeldir', type=str, default='modelAugment',
		help='model directory')
	parser.add_argument('--data_dir', dest='data_dir', type=str, default='/hdd/wsi_fun/ImageAugCustom/AugmentationOutput',
		help='data directory')
	parser.add_argument('--gpu', dest='gpu', type=str, default='0',
		help='specify which GPU to use')
	parser.add_argument('--num_steps', dest='num_steps', type=int, default=100000,
		help='maximum number of iterations')
	parser.add_argument('--save_interval', dest='save_interval', type=int, default=15000,
		help='number of iterations for saving and visualization')
	parser.add_argument('--random_seed', dest='random_seed', type=int, default=1234,
		help='random seed for initialization')
	parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.0005,
		help='weight decay')
	parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=2.5e-4,
		help='learning rate')
	parser.add_argument('--power', dest='power', type=float, default=0.9,
		help='power for polynomial decay')
	parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
		help='momentum for SGD')
	parser.add_argument('--encoder_name', dest='encoder_name', type=str, default='deeplab',
		help='name of the pre-trained model, res101, res50 or deeplab')
	parser.add_argument('--pretrain_file', dest='pretrain_file', type=str, default='deeplab_resnet.ckpt',
		help='pre-trained model filename corresponding to encoder_name')
	parser.add_argument('--data_list', dest='data_list', type=str, default='./dataAugment/train.txt',
		help='training data list filename')

	parser.add_argument('--valid_step', dest='valid_step', type=int, default=217000,
		help='number of iterations for validation')
	parser.add_argument('--valid_num_steps', dest='valid_num_steps', type=int, default=81605,
		help='number of validation samples')
	parser.add_argument('--valid_data_list', dest='valid_data_list', type=str, default='./dataAugment/val.txt',
		help='validation data list filename')
	
	parser.add_argument('--batch_size', dest='batch_size', type=int, default=15,
		help='training batch size')
	parser.add_argument('--input_height', dest='input_height', type=int, default=256,
		help='input image height')
	parser.add_argument('--input_width', dest='input_width', type=int, default=256,
		help='input image width')
	parser.add_argument('--num_classes', dest='num_classes', type=int, default=2,
		help='number of classes in images')
	parser.add_argument('--ignore_label', dest='ignore_label', type=int, default=255,
		help='label to ignore during training')
	parser.add_argument('--random_scale', dest='random_scale', type=bool, default=False,
		help='whether to perform random scaling data-augmentation')
	parser.add_argument('--random_mirror', dest='random_mirror', type=bool, default=False,
		help='whether to perform random left-right flipping data-augmentation')
	parser.add_argument('--print_color', dest='print_color', type=str, default="\033[1;32;40m",
		help='color code for printed output')

	parser.add_argument('--logfile', dest='logfile', type=str, default='log.txt',
		help='training log file name')
	parser.add_argument('--logdir', dest='logdir', type=str, default='log',
		help='traning log directory')

	args = parser.parse_args()

	# Call main and exit with proper return code
	exit_code = main(args)
	if exit_code != 0:
		exit(exit_code)
