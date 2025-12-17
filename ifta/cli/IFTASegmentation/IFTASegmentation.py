import os
import sys
from subprocess import call
from girder_client import GirderClient
from ctk_cli import CLIArgumentParser

sys.path.append("..")

def print_args(args):
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("\n")

def get_girder_client(args):
    gc = GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)
    return gc

def get_image_info(gc, input_file):
    file_info = gc.getFile(input_file)
    return file_info

def download_image(gc, input_image_info, base_dir):
    input_id = input_image_info['_id']
    input_name = input_image_info['name']
    local_input_path = "{}/{}".format(base_dir, input_name)
    try:
        gc.downloadFile(input_id, local_input_path)
        print(f"Downloaded {input_name} files to {local_input_path}")
        return local_input_path
    except Exception as e:
        print(f"Error downloading file {input_name}: {e}")

def main(args):
    print(sys.executable)

    print_args(args)

    project_name = args.base_dir.split('/')[-1]
    girder_folder_id = args.base_dir.split('/')[-2]
    print("Project name: {}".format(project_name))

    if os.path.exists("/mnt/girder_worker"):
        print("Using /mnt/girder_worker as working directory")
        base_dir = '{}/{}'.format('/mnt/girder_worker', os.listdir('/mnt/girder_worker')[0])
    else:
        print("Using /tmp/ as working directory")
        base_dir = os.getenv("TMPDIR")

    gc = get_girder_client(args)
    input_image_info = get_image_info(gc, args.input_file)
    input_file = input_image_info['name']
    file_ext = str(os.path.splitext(input_file)[1].lower())
    if file_ext not in ['.tif', '.svs']:
        raise ValueError("Unsupported file format: {}. Only .tif and .svs are supported.".format(file_ext))
    
    input_path = download_image(gc, input_image_info, base_dir)
    output_annotation_name = args.output_annotation_name.replace(" ", "_")

    print("Input file: {}".format(input_file))
    cmd = "python ../ifta_code/segmentation_school.py --project {} --option {} --base_dir {} --model {} --boxSizeHR {} --overlap_percentHR {} --classNum {} --one_network {} --encoder_name {} --wsi_ext {} --girderApiUrl {} --girderToken {} --input_file \'{}\' --input_path '{}' --girderFolderId {} --output_annotation_name {}".format(
        project_name,
        'predict',
        base_dir,
        args.model,
        args.boxSizeHR,
        args.overlap_percentHR,
        4, 
        'True', 
        'deeplab', 
        file_ext,
        args.girderApiUrl, 
        args.girderToken, 
        input_file,
        input_path,
        girder_folder_id,
        output_annotation_name
    )
    print(cmd)
    sys.stdout.flush()
    rtn_code = call(cmd, shell=True)


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
