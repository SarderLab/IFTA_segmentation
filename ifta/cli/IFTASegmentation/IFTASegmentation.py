import os
import sys
from ctk_cli import CLIArgumentParser


sys.path.append("..")


def main(args):  
    
    print(os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'])
    cmd = "python3 ../ifta_code/segmentation_school.py --option {}  --basedir {}  --model {} --boxSizeHR {} --overlap_percentHR {} --classNum {} --one_network {} --encoder_name {} --girderApiUrl {} --girderToken {} \
             --input_files {}".format('predict', args.basedir, args.model, args.boxSizeHR, args.overlap_percentHR, 4, 'True', 'deeplab', args.girderApiUrl, args.girderToken, args.input_files)
    print(cmd)
    sys.stdout.flush()
    os.system(cmd)  


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())