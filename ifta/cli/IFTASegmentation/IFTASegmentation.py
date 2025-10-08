import os
import sys
from ctk_cli import CLIArgumentParser

sys.path.append("..")

"""
    python3 ../ifta_code/segmentation_school.py 
        --option predict 
        --basedir /mnt/girder_worker/9df7a659f13f4508852b2c8c51d00358/68e5218da26c8afcfcdb1666/IFTA_test 
        --model /mnt/girder_worker/9df7a659f13f4508852b2c8c51d00358/68e55e00a26c8afcfcdbdb1e/ifta_tf2 
        --boxSizeHR 3000 
        --overlap_percentHR 0.5 
        --classNum 4 
        --one_network True 
        --encoder_name deeplab 
        --girderApiUrl http://girder:8080/api/v1/ 
        --girderToken 2mSlFtzhcWvmiUqTv0OvXRWTpbcTXubbGYBF9DmOkafNufM312ZJq6gHeGDVSstS 
        --input_files /mnt/girder_worker/9df7a659f13f4508852b2c8c51d00358/UM00083_PAS.svs
"""

def print_args(args):
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("\n")



def main(args):  
    print_args(args)

    base_dir = str(args.basedir)
    project_dir = os.path.join(base_dir, args.project)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    model_dir = str(args.model)
    if not os.path.exists(model_dir):
        raise ValueError("Model directory does not exist: {}".format(model_dir))

    print("Model files in directory:")
    for model_file in os.listdir(model_dir):
        print(model_file)
    
    cmd = "python3 ../ifta_code/segmentation_school.py --option {} --basedir {} --model {} --boxSizeHR {} --overlap_percentHR {} --classNum {} --one_network {} --encoder_name {} --girderApiUrl {} --girderToken {} --input_files {}".format(
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
                    args.input_files
                )
    print(cmd)
    sys.stdout.flush()
    os.system(cmd)


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())