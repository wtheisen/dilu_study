import os, sys, json, tqdm
import argparse

from index_class import Index

def load_paths_from_file(file_path):
    with open(file_path, 'r') as f:
        img_path_list = f.read().splitlines()

    return img_path_list

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--CacheDir', default=None, help = 'Where to write index cache files')
    # parser.add_argument('--OutputDir', default=None, help = 'Where to write query results')
    # parser.add_argument('--Recall', default=10, help='How many results to retrieve')
    # parser.add_argument('--BuildIndexList', default=None)
    # parser.add_argument('--QueryIndexList', default=None)
    # args = parser.parse_args()

    iindex = Index(cache_dir='./', gpu=True)
    index_image_list = load_paths_from_file(args.BuildIndexList)

    if args.BuildIndexList:
        index.train_index(index_image_list)
        index.add_to_index(index_image_list)

    if args.QueryIndexList:
        query_image_list = load_paths_from_file(args.QueryIndexList)
        distances, ids = index.query_index(index_image_list)

        for q_d, q_i in zip(distances, ids):
            print(index.query_result_to_image(q_i))

        # print(index.queries_to_json(raw_query_output))
