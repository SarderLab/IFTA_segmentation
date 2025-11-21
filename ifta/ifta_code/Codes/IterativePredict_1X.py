import cv2
import numpy as np
import os
import sys
import multiprocessing
import lxml.etree as ET
import warnings
import time
from PIL import Image
from glob import glob
from subprocess import call
from joblib import Parallel, delayed
from skimage.io import imread
import imageio
from skimage.transform import resize
from shutil import rmtree
import json
import girder_client
from pathlib import Path

sys.path.append(os.getcwd()+'/Codes')

# Try absolute import first, then relative import
try:
    from xml_to_mask import get_num_classes
    from get_choppable_regions import get_choppable_regions
    from get_network_performance import get_perf
    from getWsi import getWsi
except ImportError:
    from .xml_to_mask import get_num_classes
    from .get_choppable_regions import get_choppable_regions
    from .get_network_performance import get_perf
    from .getWsi import getWsi

"""
Pipeline code to segment regions from WSI

"""

# define xml class colormap
xml_color = [65280, 65535, 255, 16711680, 33023]

def validate(args):
    # define folder structure dict
    dirs = {'outDir': args.base_dir + '/' + args.project + args.outDir}
    dirs['txt_save_dir'] = '/txt_files/'
    dirs['img_save_dir'] = '/img_files/'
    dirs['mask_dir'] = '/wsi_mask/'
    dirs['chopped_dir'] = '/'
    dirs['save_outputs'] = args.save_outputs
    dirs['modeldir'] = '/MODELS/'
    dirs['training_data_dir'] = '/TRAINING_data/'
    dirs['validation_data_dir'] = '/HOLDOUT_data/'

    # find current iteration
    if args.iteration == 'none':
        iteration = get_iteration(args=args)
    else:
        iteration = int(args.iteration)

    # get all WSIs
    WSIs = []
    for ext in [args.wsi_ext]:
        WSIs.append(glob(args.base_dir + '/' + args.project + dirs['validation_data_dir'] + '/*' + ext))

    if iteration == 'none':
        print('ERROR: no trained models found \n\tplease use [--option train]')

    else:
        for iter in range(1,iteration+1):
            dirs['xml_save_dir'] = args.base_dir + '/' + args.project + dirs['validation_data_dir'] + str(iter) + '_Predicted_XMLs/'


            # check main directory exists
            make_folder(dirs['outDir'])

            if not os.path.exists(dirs['xml_save_dir']):
                make_folder(dirs['xml_save_dir'])

            print('working on iteration: ' + str(iter))

            with open(args.base_dir + '/' + args.project + dirs['validation_data_dir'] + 'validation_stats.txt', 'a') as f:
                f.write('\niteration: \t'+str(iter)+'\n')
                f.write('\twsi\t\t\tsensitivity\t\t\tspecificity\t\t\tprecision\t\t\taccuracy\t\t\tprediction time\n')

            for wsi in WSIs:
                # predict xmls
                startTime = time.time()

                filename=dirs['xml_save_dir']+'/'+ (wsi.split('/')[-1]).split('.')[0] +'.xml'
                if not os.path.isfile(filename):
                    predict_xml(args=args, dirs=dirs, wsi=wsi, iteration=iter)

                predictTime = time.time() - startTime
                # test performance
                gt_xml = os.path.splitext(wsi)[0] + '.xml'
                predicted_xml = gt_xml.split('/')
                predicted_xml = dirs['xml_save_dir'] + predicted_xml[-1]
                sensitivity,specificity,precision,accuracy = get_perf(wsi=wsi, xml1=gt_xml, xml2 = predicted_xml, args=args)

                with open(args.base_dir + '/' + args.project + dirs['validation_data_dir'] + 'validation_stats.txt', 'a') as f:
                    f.write('\t'+wsi.split('/')[-1]+'\t\t'+str(sensitivity)+'\t\t'+str(specificity)+'\t\t'+str(precision)+'\t\t'+str(accuracy)+'\t\t'+str(predictTime)+'\n')

        print('\n\n\033[92;5mDone validating: \n\t\033[0m\n')

def predict(args):
    # define folder structure dict
    dirs = {'outDir': args.base_dir + '/' + args.project + args.outDir}
    dirs['txt_save_dir'] = '/txt_files/'
    dirs['img_save_dir'] = '/img_files/'
    dirs['mask_dir'] = '/wsi_mask/'
    dirs['chopped_dir'] = '/'
    dirs['save_outputs'] = args.save_outputs
    dirs['training_data_dir'] = '/TRAINING_data/'
    dirs['logging_dir'] = args.base_dir + '/' + args.project + '/LOGS/'

    # find current iteration
    if args.iteration == 'none':
        iteration = get_iteration(args=args)
    else:
        iteration = int(args.iteration)

    dirs['xml_save_dir'] = args.base_dir + '/' + args.project + dirs['training_data_dir'] + str(iteration-1) + '/Predicted_XMLs/'

    if iteration == 'none':
        print('ERROR: no trained models found \n\tplease use [--option train]')

    else:
        # check main directory exists
        make_folder(dirs['outDir'])
        make_folder(dirs['xml_save_dir'])

        # get all WSIs
        for name, path in zip([args.input_file], [args.input_path]):
            if os.path.exists(path) and os.path.isfile(path):
                predict_xml(args=args, dirs=dirs, wsi={"name": name, "path": path}, iteration=iteration)


def predict_xml(args, dirs, wsi, iteration):
    # reshape regions calc
    downsample = int(args.downsampleRateHR**.5)
    region_size = int(args.boxSizeHR*(downsample))
    step = int(region_size*(1-args.overlap_percentHR))

    # figure out the number of classes
    if args.classNum == 0:
        annotatedXMLs = glob(args.base_dir + '/' + args.project + dirs['training_data_dir'] + str(iteration-1) + '/*.xml')
        classes = []
        for xml in annotatedXMLs:
            classes.append(get_num_classes(xml))
        classNum = max(classes)
    else:
        classNum = args.classNum

    # chop wsi
    if args.chop_data == 'True':
        # chop wsi
        fileID, test_num_steps = chop_suey(wsi, dirs, downsample, region_size, step, args)
        dirs['fileID'] = fileID
        print('Chop SUEY!\n')
    else:
        slide = getWsi(wsi['path'])
        # get image dimensions
        dim_x, dim_y = slide.dimensions
        fileID = wsi['name'].split('.')[-2]
        dirs['fileID'] = fileID = fileID.replace(' ', '_')
        test_num_steps = file_len(dirs['outDir'] + fileID + dirs['txt_save_dir'] + fileID + '_images' + ".txt")

    # call DeepLab for prediction
    print('Segmenting tissue ...\n')

    make_folder(dirs['outDir'] + fileID + dirs['img_save_dir'] + 'prediction')

    test_data_list = fileID + '_images' + '.txt'
    modeldir = args.model
    
    # Debug: Show the exact command being run
    batch_size = args.batch_size
    num_batches = (test_num_steps + batch_size - 1) // batch_size  # Ceiling division

    # Construct the DeepLab command
    deeplab_cmd = ['python', '../ifta_code/Codes/Deeplab_network/main.py',
        '--option', 'predict',
        '--test_data_list', dirs['outDir']+fileID+dirs['txt_save_dir']+test_data_list,
        '--out_dir', dirs['outDir']+fileID+dirs['img_save_dir'],
        '--test_num_steps', str(num_batches),  # Use number of batches, not number of images
        '--modeldir', modeldir,
        '--data_dir', dirs['outDir']+fileID+dirs['img_save_dir'],
        '--num_classes', str(classNum),
        '--gpu', str(args.gpu),
        '--encoder_name',args.encoder_name,
        '--batch_size', str(batch_size),
        '--logdir', dirs['logging_dir']
    ]

    print("DeepLab command:", ' '.join(deeplab_cmd))
    print(f"Processing {test_num_steps} images in {num_batches} batches of {batch_size}")
    return_code = call(deeplab_cmd)
    
    # Check if prediction was successful
    if return_code != 0:
        print(f"ERROR: DeepLab prediction failed with return code {return_code}")
        return
    
    # Check how many mask files were generated
    prediction_dir = dirs['outDir'] + fileID + dirs['img_save_dir'] + 'prediction'
    if os.path.exists(prediction_dir):
        mask_files = glob(prediction_dir + '/*_mask.png')
        print(f"Generated {len(mask_files)} mask files out of {test_num_steps} expected")
        if len(mask_files) == 0:
            print("ERROR: No mask files were generated!")
            return
    else:
        print("ERROR: Prediction directory was not created!")
        return

    # un chop
    print('\nreconstructing wsi map ...\n')
    wsiMask = un_suey(dirs=dirs, args=args)

    # save hotspots
    if dirs['save_outputs'] == True:
    	#reduce the resolution of the image
        wsidims = wsiMask.shape
        wsiMask_save = resize(wsiMask,(int(wsidims[0]/4),int(wsidims[1]/4)),order=0,preserve_range=True)

        make_folder(dirs['outDir'] + fileID + dirs['mask_dir'])
        print('saving to: ' + dirs['outDir'] + fileID + dirs['mask_dir'] + fileID  + '.png')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imageio.imwrite(dirs['outDir'] + fileID + dirs['mask_dir'] + fileID + '.png', wsiMask_save)

    print('\n\nStarting XML construction: ')

    xml_suey(wsiMask=wsiMask, args=args, downsample=downsample, glob_offset=[0,0])

    # clean up
    if dirs['save_outputs'] == False:
        print('cleaning up')
        rmtree(dirs['outDir']+fileID)


def get_iteration(args):
    currentmodels=os.listdir(args.base_dir + '/' + args.project + '/MODELS/')

    if not currentmodels:
        return 'none'
    else:
        currentmodels=map(int,currentmodels)
        Iteration=np.max(list(currentmodels))
        return Iteration

def get_test_step(modeldir):
    pretrains=glob(modeldir + '/*.ckpt*')

    maxmodel=0
    for modelfiles in pretrains:
        modelID=modelfiles.split('.')[-2].split('-')[1]
        try:
            modelID = int(modelID)
            if modelID>maxmodel:
                maxmodel=modelID
        except: pass

    return maxmodel

def make_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) # make directory if it does not exit already # make new directory

def restart_line(): # for printing chopped image labels in command line
    sys.stdout.write('\r')
    sys.stdout.flush()

def file_len(fname): # get txt file length (number of lines)
    with open(fname) as f:
        for i, l in enumerate(f):
            pass

    if 'i' in locals():
        return i + 1

    else:
        return 0


def chop_suey(wsi, dirs, downsample, region_size, step, args): # chop wsi
    print('\nopening: ' + wsi['name'])

    slide = getWsi(wsi['path'])
    # get image dimensions
    dim_x, dim_y = slide.dimensions

    fileID=wsi['name'].split('.')[-2]
    dirs['fileID'] = fileID = fileID.replace(' ', '_')
    print('\nchopping ...\n')

    # make txt file
    make_folder(dirs['outDir'] + fileID + dirs['txt_save_dir'])
    f_name = dirs['outDir'] + fileID + dirs['txt_save_dir'] + fileID + ".txt"
    f2_name = dirs['outDir'] + fileID + dirs['txt_save_dir'] + fileID + '_images' + ".txt"
    f = open(f_name, 'w')
    f2 = open(f2_name, 'w')
    f2.close()

    make_folder(dirs['outDir'] + fileID + dirs['img_save_dir'] + dirs['chopped_dir'])

    f.write('Image dimensions:\n')

    # make index for iters
    index_y=np.array(range(0,dim_y,step))
    index_x=np.array(range(0,dim_x,step))
    index_y[-1]=dim_y-step
    index_x[-1]=dim_x-step

    f.write('X dim: ' + str((index_x[-1]+region_size)/downsample) +'\n')
    f.write('Y dim: ' + str((index_y[-1]+region_size)/downsample) +'\n\n')
    f.write('Regions:\n')
    f.write('image:xStart:xStop:yStart:yStop\n\n')
    f.close()

    # get non white regions
    choppable_regions = get_choppable_regions(wsi=wsi, index_x=index_x, index_y=index_y, boxSize=region_size,white_percent=args.white_percent)

    print('saving region:')

    num_cores = multiprocessing.cpu_count()

    Parallel(n_jobs=num_cores, backend='threading')(delayed(chop_wsi)(yStart=i, xStart=j, idxx=idxx, idxy=idxy, f_name=f_name, f2_name=f2_name, dirs=dirs, downsample=downsample, region_size=region_size, args=args, wsi=wsi, choppable_regions=choppable_regions) for idxy, i in enumerate(index_y) for idxx, j in enumerate(index_x))

    test_num_steps = file_len(dirs['outDir'] + fileID + dirs['txt_save_dir'] + fileID + '_images' + ".txt")
    print('\n\t' + str(test_num_steps) +' image regions chopped')

    return fileID, test_num_steps

def chop_wsi(yStart, xStart, idxx, idxy, f_name, f2_name, dirs, downsample, region_size, args, wsi, choppable_regions): # perform cutting in parallel
    if choppable_regions[idxy, idxx] != 0:
        yEnd = yStart+region_size
        xEnd = xStart+region_size
        xLen=xEnd-xStart
        yLen=yEnd-yStart

        slide = getWsi(wsi['path'])
        tile = slide.read_region((xStart, yStart), level=0, size=(region_size, region_size))
        tile = tile.convert("RGB")
        subsect = np.array(tile, dtype=np.uint8)

        imageIter = str(xStart)+str(yStart)

        f = open(f_name, 'a+')
        f2 = open(f2_name, 'a+')

        # append txt file
        f.write(imageIter + ':' + str(xStart/downsample) + ':' + str(xEnd/downsample)
            + ':' + str(yStart/downsample) + ':' + str(yEnd/downsample) + '\n')

		# resize images ans masks
        if downsample > 1:
            c=(subsect.shape)
            s1=int(c[0]/downsample)
            s2=int(c[1]/downsample)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                subsect=resize(subsect,(s1,s2), mode='constant')

        # save image
        directory = dirs['outDir'] + dirs['fileID'] + dirs['img_save_dir'] + dirs['chopped_dir']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if np.issubdtype(subsect.dtype, np.floating):
                subsect = (np.clip(subsect, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(directory + dirs['fileID'] + str(imageIter) + args.imBoxExt,subsect)

        f2.write(dirs['fileID'] + str(imageIter) + args.imBoxExt + '\n')
        f.close()
        f2.close()

def un_suey(dirs, args): # reconstruct wsi from predicted masks
    txtFile = dirs['fileID'] + '.txt'

    # read txt file
    f = open(dirs['outDir'] + dirs['fileID'] + dirs['txt_save_dir'] + txtFile, 'r')
    lines = f.readlines()
    f.close()
    lines = np.array(lines)

    # get wsi size
    xDim =int(float((lines[1].split(': ')[1]).split('\n')[0]))
    yDim = int(float((lines[2].split(': ')[1]).split('\n')[0]))

    # make wsi mask
    wsiMask = np.zeros([yDim, xDim]).astype(np.uint8)

    # read image regions
    for regionNum in range(7, np.size(lines)):
        # get region
        region = lines[regionNum].split(':')
        region[4] = region[4].split('\n')[0]

        # read mask - skip if file doesn't exist
        mask_path = dirs['outDir'] + dirs['fileID'] + dirs['img_save_dir'] + 'prediction/' + dirs['fileID'] + region[0] + '_mask.png'
        if not os.path.exists(mask_path):
            print(f'\nWarning: Mask file not found, skipping region {region[0]}: {mask_path}')
            continue
            
        mask = imread(mask_path)

        # get region bounds
        xStart = np.uint32(float(region[1]))
        xStop = np.uint32(float(region[2]))
        yStart = np.uint32(float(region[3]))
        if yStart < 0:
            yStart = 0
        yStop = np.uint32(float(region[4]))

        # Calculate expected region size and resize mask if necessary
        expected_height = yStop - yStart
        expected_width = xStop - xStart
        if mask.shape != (expected_height, expected_width):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mask = resize(mask, (expected_height, expected_width), order=0, preserve_range=True).astype(np.uint8)

        mask_part = wsiMask[yStart:yStop, xStart:xStop]
        ylen, xlen = np.shape(mask_part)
        mask = mask[:ylen, :xlen]

        # populate wsiMask with max
        wsiMask[yStart:yStop, xStart:xStop] = np.maximum(mask_part, mask).astype(np.uint8)

    return wsiMask

def xml_suey(wsiMask, args, downsample, glob_offset):
    # make xml
    Annotations = xml_create()
    annotation_index = 1
    # add annotation
    Annotations = xml_add_annotation(Annotations=Annotations, annotationID=annotation_index)

    unique_mask = []
    for i in range(0, len(wsiMask), 7000):
        unique_mask.extend(np.unique(wsiMask[i:i + 7000]))

    print(np.unique(wsiMask))
    
    # print output
    print('\t Working on: annotationID ' + str(annotation_index))
    # get only 1 class binary mask
    binary_mask = np.zeros(np.shape(wsiMask)).astype('uint8')
    binary_mask[wsiMask == annotation_index] = 1
    print('Binary_mask ==', np.unique(binary_mask))

    # add mask to xml
    pointsList = get_contour_points(binary_mask, args=args, downsample=downsample,value=annotation_index,offset={'X':glob_offset[0],'Y':glob_offset[1]})
    for i in range(len(pointsList)):
        pointList = pointsList[i]
        Annotations = xml_add_region(Annotations=Annotations, pointList=pointList, annotationID=annotation_index)

    # save xml
    upload_to_girder(args, Annotations)

def upload_to_girder(args, Annotations):
    print(f"Using data from girder_client Folder: {args.project}")

    file_name = args.input_file
    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)

    files = list(gc.listItem(args.girderFolderId))
    item_dict = dict()
    for file in files:
        d = {file['name']: file['_id']}
        item_dict.update(d)

    print(item_dict)

    print('uploading annotation to girder...')
    annots = convert_xml_json(Annotations, [args.output_annotation_name])
    for annot in annots:
        _ = gc.post(path='annotation', parameters={'itemId': item_dict[file_name]}, data=json.dumps(annot))

    print(f'annotation {args.output_annotation_name} uploaded...\n')

def get_contour_points(mask, args, downsample, value, offset={'X': 0,'Y': 0}):
    # returns a dict pointList with point 'X' and 'Y' values
    # input greyscale binary image
    maskPoints, contours = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    pointsList = []
    
    for j in np.array(range(len(maskPoints))):
        if len(maskPoints[j])>2:
            if cv2.contourArea(maskPoints[j]) > args.min_size:
                pointList = []
                for i in np.array(range(0,len(maskPoints[j]),4)):
                    point = {'X': (maskPoints[j][i][0][0] * downsample) + offset['X'], 'Y': (maskPoints[j][i][0][1] * downsample) + offset['Y']}
                    pointList.append(point)
                pointsList.append(pointList)
    return pointsList


### functions for building an xml tree of annotations ###
def xml_create(): # create new xml tree
    # create new xml Tree - Annotations
    Annotations = ET.Element('Annotations')
    return Annotations

def xml_add_annotation(Annotations, annotationID=None): # add new annotation
    # add new Annotation to Annotations
    # defualts to new annotationID
    if annotationID == None: # not specified
        annotationID = len(Annotations.findall('Annotation')) + 1
    Annotation = ET.SubElement(Annotations, 'Annotation', attrib={'Type': '4', 'Visible': '1', 'ReadOnly': '0', 'Incremental': '0', 'LineColorReadOnly': '0', 'LineColor': str(xml_color[annotationID-1]), 'Id': str(annotationID), 'NameReadOnly': '0'})
    Regions = ET.SubElement(Annotation, 'Regions')
    return Annotations

def xml_add_region(Annotations, pointList, annotationID=-1, regionID=None): # add new region to annotation
    # add new Region to Annotation
    # defualts to last annotationID and new regionID
    Annotation = Annotations.find("Annotation[@Id='" + str(annotationID) + "']")
    Regions = Annotation.find('Regions')
    if regionID == None: # not specified
        regionID = len(Regions.findall('Region')) + 1
    Region = ET.SubElement(Regions, 'Region', attrib={'NegativeROA': '0', 'ImageFocus': '-1', 'DisplayId': '1', 'InputRegionId': '0', 'Analyze': '0', 'Type': '0', 'Id': str(regionID)})
    Vertices = ET.SubElement(Region, 'Vertices')
    for point in pointList: # add new Vertex
        ET.SubElement(Vertices, 'Vertex', attrib={'X': str(point['X']), 'Y': str(point['Y']), 'Z': '0'})
    # add connecting point
    ET.SubElement(Vertices, 'Vertex', attrib={'X': str(pointList[0]['X']), 'Y': str(pointList[0]['Y']), 'Z': '0'})
    return Annotations

def xml_save(Annotations, filename):
    xml_data = ET.tostring(Annotations, pretty_print=True)
    #xml_data = Annotations.toprettyxml()
    f = open(filename, 'wb')
    f.write(xml_data)
    f.close()

def read_xml(filename):
    # import xml file
    tree = ET.parse(filename)
    root = tree.getroot()

def convert_xml_json(root, names, colorList=None, alpha=0.4):
    
    if colorList == None:
        colorList = ["rgb(0, 255, 128)", "rgb(0, 255, 255)", "rgb(255, 255, 0)", "rgb(255, 128, 0)", "rgb(0, 128, 255)",
                     "rgb(0, 0, 255)", "rgb(0, 102, 0)", "rgb(153, 0, 0)", "rgb(0, 153, 0)", "rgb(102, 0, 204)",
                     "rgb(76, 216, 23)", "rgb(102, 51, 0)", "rgb(128, 128, 128)", "rgb(0, 153, 153)", "rgb(0, 0, 0)"]
        
    anns = root.findall('Annotation')
    assert len(anns) <= len(names)

    data = []
    for n, child in enumerate(anns):
        dataDict = dict()
        name = names[n]
        print(f"Building JSON layer: [{name}]")
        element = []
        reg = child.find('Regions')
        for i in reg.findall('Region'):
            eleDict = dict()
            eleDict["closed"] = True

            lineColor = colorList[n % len(colorList)]
            eleDict["lineColor"] = lineColor

            fillColor = lineColor[:3]+'a'+lineColor[3:-1] + f', {alpha})'
            eleDict["fillColor"] = fillColor

            eleDict["lineWidth"] = 2
            points = []
            ver = i.find('Vertices')
            Verts = ver.findall('Vertex')
            if len(Verts) <= 1:
                continue  # skip if only 1 vertex points
            for j in Verts:
                eachPoint = []
                eachPoint.append(float(j.get('X')))
                eachPoint.append(float(j.get('Y')))
                eachPoint.append(float(j.get('Z')))
                points.append(eachPoint)
            eleDict["points"] = points
            eleDict["type"] = "polyline"
            element.append(eleDict)
        dataDict["elements"] = element
        dataDict["name"] = name
        data.append(dataDict)

    return data    
