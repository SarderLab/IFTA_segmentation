import os
import sys
from subprocess import call
from ctk_cli import CLIArgumentParser

sys.path.append("..")

def print_args(args):
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("\n")



def main(args):
    print(sys.executable)

    print_args(args)

    cmd = "python ../ifta_code/segmentation_school.py --option {} --basedir {} --model {} --boxSizeHR {} --overlap_percentHR {} --classNum {} --one_network {} --encoder_name {} --girderApiUrl {} --girderToken {} --input_files {} --output_annotation_name {}".format(
                    'predict',
                    args.basedir,
                    args.model,
                    args.boxSizeHR,
                    args.overlap_percentHR,
                    4, 
                    'True', 
                    'deeplab', 
                    args.girderApiUrl, 
                    args.girderToken, 
                    args.input_files,
                    args.annot_name
                )
    print(cmd)
    sys.stdout.flush()
    rtn_code = call(cmd, shell=True)


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())