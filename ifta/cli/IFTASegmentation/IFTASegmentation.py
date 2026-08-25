import os
import sys
from subprocess import call
from storage_client import StorageClient

sys.path.append("..")


def main():
    item_id = os.environ['ITEM_ID']
    storage_api_url = os.environ['STORAGE_API_URL']
    job_auth_token = os.environ['JOB_AUTH_TOKEN']
    model_id = os.environ['MODEL_ID']
    box_size_hr = os.environ.get('BOX_SIZE_HR', '3000')
    overlap_percent_hr = os.environ.get('OVERLAP_PERCENT_HR', '0.5')
    batch_size = min(int(os.environ.get('BATCH_SIZE', '2')), 16)
    output_annotation_name = os.environ.get('OUTPUT_ANNOTATION_NAME', 'ifta').replace(' ', '_')

    if os.path.exists("/mnt/girder_worker"):
        print("Using /mnt/girder_worker as working directory")
        base_dir = '{}/{}'.format('/mnt/girder_worker', os.listdir('/mnt/girder_worker')[0])
    else:
        print("Using /tmp/ as working directory")
        base_dir = os.getenv("TMPDIR", "/tmp")
    os.makedirs(base_dir, exist_ok=True)

    project_name = base_dir.split('/')[-1]
    print("Project name: {}".format(project_name))

    client = StorageClient(storage_api_url, job_auth_token)

    input_file = f'{item_id}.svs'
    input_path = os.path.join(base_dir, input_file)
    print(f'Downloading input for item {item_id} to {input_path}')
    client.download_input(item_id, input_path)

    file_ext = str(os.path.splitext(input_file)[1].lower())
    if file_ext not in ['.tif', '.svs']:
        raise ValueError("Unsupported file format: {}. Only .tif and .svs are supported.".format(file_ext))

    model_dir = os.path.join(base_dir, 'model')
    print(f'Downloading model {model_id} to {model_dir}')
    client.download_model_dir(model_id, model_dir)

    print("Input file: {}".format(input_file))
    cmd = "python ../ifta_code/segmentation_school.py --project {} --option {} --model {} --boxSizeHR {} --overlap_percentHR {} --classNum {} --one_network {} --encoder_name {} --wsi_ext {} --storage_api_url {} --job_auth_token {} --item_id {} --input_file '{}' --input_path '{}' --output_annotation_name {}".format(
        project_name,
        'predict',
        model_dir,
        box_size_hr,
        overlap_percent_hr,
        4,
        'True',
        'deeplab',
        file_ext,
        storage_api_url,
        job_auth_token,
        item_id,
        input_file,
        input_path,
        output_annotation_name,
    )

    cmd += ' --batch_size {}'.format(batch_size)

    print(cmd)
    sys.stdout.flush()
    rtn_code = call(cmd, shell=True)
    if rtn_code != 0:
        raise RuntimeError(f"Segmentation pipeline failed with exit code {rtn_code}")


if __name__ == "__main__":
    main()
