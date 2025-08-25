import os
import glob
import argparse
import girder_client

def diff_remaining_slides(path):
    slides = list(map(lambda x: os.path.basename(x).split(".")[0], glob.glob(os.path.join(path, "*.svs"))))
    xmls = list(map(lambda x: os.path.basename(x).split(".")[0], glob.glob(os.path.join(path, "Predicted_XMLs", "*.xml"))))

    difference = set(slides) - set(xmls)
    print("Slides without corresponding XMLs:", len(difference))
    return difference
     
def delete_all_except(path, diff_slides):
    all_files = os.listdir(path)
    
    for idx, file_name in enumerate(all_files):
        if file_name.split(".")[0] not in diff_slides:
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted {idx + 1}: {file_path}")

def delete_annotation(gc, annotation, item_name: str, annotation_names: list):
    if annotation['annotation']['name'] in annotation_names:
        print(f"Deleting annotation for item: {item_name}")
        gc.delete('/annotation/{}'.format(annotation['_id']))

def download_annotation(gc, annot, annot_to_download: str, item_name: str, destination_path: str):
    if annot['annotation']['name'] == annot_to_download:
        annotation = gc.get('/annotation/{}'.format(annot['_id']))
        annotation_file = os.path.join(destination_path, f"{item_name}_annotation.json")
        with open(annotation_file, 'w') as f:
            f.write(str(annotation))
        print(f"Saved annotation for item: {item_name} to {annotation_file}")

def get_relevant_annotations(gc, folder_id, destination_path="annotations/"):
    """
    Get relevant annotations from Girder.
    """
    # Get all items in the folder
    os.makedirs(destination_path, exist_ok=True)
    items = gc.listItem(folder_id)
    for item in items:
        # Get the item ID
        item = gc.getItem(item['_id'])
        # Get the item name
        item_name = item['name']

        # Get the item annotations
        annotations = gc.get('/annotation/item/{}'.format(item['_id']), parameters={'sort': 'updated'})
        annotations.reverse()
        annotations = list(annotations)

        for annot in annotations:
            # delete_annotation(gc, annot, item_name, ["tubules", "globally_sclerotic_glomeruli"])
            download_annotation(gc, annot, "non_globally_sclerotic_glomeruli", item_name, destination_path)

def main(args):
    # gc = girder_client.GirderClient(apiUrl=args.girder_url)
    # gc.setToken(args.girder_token)
    # print("Girder client initialized.")
    
    # get_relevant_annotations(gc, args.folder_id)
    # print("Finished processing annotations.")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test IFTA segmentation.")
    parser.add_argument(
        "--girder-url",
        type=str,
        default="https://athena.rc.ufl.edu/api/v1",
        help="URL of the Girder server.",
    )
    parser.add_argument(
        "--girder-token",
        type=str,
        default="Zs1B7csT05Icu59hlUtcEWLnWNAVIpZ4saQwMxYq8zkjfNuQx7HSWiu7kPlinjSe",
        help="API token for Girder server.",
    )
    parser.add_argument(
        "--folder-id",
        type=str,
        default="67f03d6f8733e17e297fa5db",
        help="ID of the folder to process.",
    )
    args = parser.parse_args()
    main(args)