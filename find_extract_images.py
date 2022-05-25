import os, sys, json
import argparse, concurrent.futures
import rawpy

from tqdm import tqdm
from image_processor import PreFeatureExtraction

finished = False
file_list = []

def find_images(directories, dataset_path_name):
    global file_list
    global finished

    fname_to_path_dict = {}
    fname_to_ID_dict = {}
    ID_to_name_dict = {}
    fcount = 0

    try:
        for d in directories:
            print("searching " + d)

            for root, dirs, files in os.walk(d, followlinks=True):
                num_length = len(dirs)

                for f in tqdm(files):
                    file_ID = '.'.join(f.split('.')[:-1])

                    if file_ID not in fname_to_path_dict.keys() and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        fname_path = os.path.join(os.path.abspath(root), f)
                        fname_to_path_dict[file_ID] = fname_path
                        file_list.append(fname_path)
                        fcount += 1
                    else:
                        print('[WARNING]: file already detected, skip...')
    except Exception as e:
        print(f'[ERROR]: Failure when searching files: {e}')

    finished = True
    output_path = os.path.join(dataset_path_name + "_pathmap.json")
    output_list_path = os.path.join(dataset_path_name + "_filelist.txt")

    print('[LOG]: Images sourced, writing map dicts...')
    with open(output_path, 'w') as f:
        json.dump(fname_to_path_dict, f)
    with open(output_list_path, 'w') as fp:
        fp.write('\n'.join(file_list))

    return fname_to_path_dict, file_list

def extract_image_features(image_path):
    image_path = image_path.rstrip()

    try:
        feature_dict = featureExtractor.process_image(image_path, tfcores=args.TFCores)

        outpath = os.path.join(args.OutputDir, f'features_{args.DatasetName}', feature_dict['supplemental_information']['value'].key)
        with open(outpath, 'wb') as of:
            of.write(feature_dict['supplemental_information']['value']._data)

        # print(f'[LOG]: Exctracted features for {image_path}')
    except rawpy._rawpy.LibRawNonFatalError as e:
        print(f"[WARNING]: Problem with {image_path}\n\t[EXCEPTION]: {e}")

def process_images(consumer=False):
    global file_list
    global finished

    try:
        os.mkdir(os.path.join(args.OutputDir, f'features_{args.DatasetName}'))
    except:
        print('[WARNING]: Failed to create feature folder (may already exist)')

    print('[LOG]: Beginning feature extraction...')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.PE) as executor:
        list(tqdm(executor.map(extract_image_features, file_list), total=len(file_list)))
    print('[LOG]: Feature extraction complete...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ImageDirectoryList', help = 'Comma separated list of image directories')
    parser.add_argument('--OutputDir', help = 'Path to output directory')
    parser.add_argument('--DatasetName', help = 'Name of the output dataset')
    parser.add_argument('--Parallel', type = int, default = 0)
    parser.add_argument('--FindOnly', type = int, default = 0)
    parser.add_argument('--InputFileList', default=None)
    parser.add_argument('--PE', type=int, default=1)
    parser.add_argument('--TFCores', type=int, default=1)
    parser.add_argument('--det', help='Keypoint Detection method', default='SURF')
    parser.add_argument('--desc', help='Feature Description method', default='SURF')
    parser.add_argument('--kmax', type=int, default=5000, help='Max keypoints per image')
    args = parser.parse_args()

    numJobs = 10

    try:
        dir_list = args.ImageDirectoryList.split(',')
    except:
        print('Failed to split on ,')
        dir_list = list(args.ImageDirectoryList)

    featureExtractor = PreFeatureExtraction()
    if args.Parallel and not args.FindOnly and not args.InputFileList:
        process_images(args.Parallel)

    if not args.InputFileList:
        name_to_path_dict, file_list = find_images(dir_list, os.path.join(args.OutputDir, args.DatasetName))

        if args.FindOnly:
            exit(0)
    else:
        with open(args.inputFileList,'r') as fp:
            file_list = fp.readlines()

    if not args.FindOnly:
        print('extracting image features')
        process_images(args.Parallel)
