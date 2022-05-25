import json, collections, os, time
import read_fvecs, find_ground_truth

import numpy as np

from tqdm import tqdm
# from itertools import chain
from statistics import mean

from index_class import Index
from feature_extractor import feature_detection_and_description

def prepare_json():
    results_json = {}
    results_json['trained_core_queries'] = {}
    results_json['dilued_core_queries'] = {}
    results_json['trained_dilu_queries'] = {}
    results_json['dilued_dilu_queries'] = {}

    results_json['g_t_deltas'] = {}

    for i in range(0, 100):
        results_json['trained_core_queries'][str(i)] = []
        results_json['dilued_core_queries'][str(i)] = []
        results_json['trained_dilu_queries'][str(i)] = []
        results_json['dilued_dilu_queries'][str(i)] = []

        results_json['g_t_deltas'][str(i)] = []

    return results_json

def mAP_metric(sorted_live, sorted_g_t, recall):
    mAP = 0
    cor_count = 0

    for i in range(0, recall):
        if sorted_live[i] == sorted_g_t[i]:
            cor_count += 1
            mAP += float(cor_count / (i + 1))

    return (1 / recall) * mAP

def feature_to_image_voting(id_to_path, returned_feature_ids, recall=10):
    s = time.time()
    voted_images = collections.Counter()

    for feature_result_ids in returned_feature_ids:
        for result_id in feature_result_ids:
            voted_images[id_to_path[result_id]] += 1

    e = time.time()
    # tqdm.write(f'Voting took {e - s}')
    return voted_images.most_common(recall)

def feature_extraction(img_path):
    keyps, feats, det_t, dsc_t = feature_detection_and_description(img_path)
    return feats

def dilute(core, destractor, dilu_amount):
    return np.concatenate([core, destractor[:dilu_amount]])

def build_gt_index(feature_vectors_in_index, ids):
    s = time.time()
    i = find_ground_truth.build_flat_index(feature_vectors_in_index, ids, dim=64)
    e = time.time()
    tqdm.write(f'Building ground-truth took {e - s}')
    return i

def query_gt(gt_index, query_feature_vectors, recall):
    s = time.time()
    dists, ids = find_ground_truth.find_ground_truth(query_feature_vectors, gt_index, recall=recall)
    e = time.time()
    # tqdm.write(f'Querying ground-truth took {e - s}')
    return ids

def walk_classes_dirs(root_path):
    class_dir_paths = []
    for i in os.listdir(root_path):
        if os.path.isdir(os.path.join(root_path, i)):
            class_dir_paths.append(os.path.join(root_path, i))

    return class_dir_paths

# with open('./surf_results_template.json') as f:
#     results_json = json.load(f)
results_json = prepare_json()

recall = 25
num_queries = 25
num_add_queries = 5
num_dilu_steps = 10
trials = 5
dataset = 'Apu'

feature_recall = 100

mAP = False
feature_level_matching = True
dilu_set_queries = True
load_from_images = True
images_root_path = '/home/wtheisen/Dropbox/4chan/downloads/Memes/Apu/'

id_to_path = {}
feature_to_id = []

training_features = []
feature_list = []
query_features = []

previous_g_t = []
g_t_change_count = 0

tqdm.write(f'Loading dataset {dataset}')

img_class_features_dict = {}

if load_from_images:
    img_count = 0

    s = time.time()
    image_path_tuple_list = {}

    for image in tqdm(os.listdir(images_root_path), desc='Images'):
        if not image.endswith('.png') and not image.endswith('.jpg'):
            continue

        feats = feature_extraction(os.path.join(images_root_path, image))

        try:
            if not feats.any():
                tqdm.write('Failed to extract features, skipping...')
                continue
        except:
            tqdm.write('Failed to extract features, skipping...')
            continue

        ided_feats = []
        for f in feats:
            ided_feats.append((f, img_count))
            feature_to_id.append((f, img_count))
        image_path_tuple_list[image] = ided_feats

        id_to_path[img_count] = image
        img_count += 1
    e = time.time()
tqdm.write(f'Loading features took {e - s}')

for i in tqdm(range(0, trials), desc='Trials'):
    #select core set from the full feature list
    image_list = list(image_path_tuple_list.keys())
    np.random.shuffle(image_list)
    rando_images = np.array_split(image_list, 4)
    rando_core_images = rando_images[0]
    # rando_dilu_images = rando_images[1]
    rando_dilu_images = np.concatenate((rando_images[1], rando_images[2], rando_images[3]), axis=None)

    rando_core_tuples = []
    for q in rando_core_images:
            rando_core_tuples += image_path_tuple_list[q]

    rando_query_images = np.random.choice(rando_core_images, num_queries)
    rando_query_images_gt = {}

    for q in rando_query_images:
        rando_query_images_gt[q] = []

    compounding_dilu_tuples = []
    rando_queries_from_dilu = []

    index_dilu = Index(gpu=True)
    index_dilu.train_index(None, training_features=np.asarray([n[0] for n in rando_core_tuples]))
    index_dilu.add_to_index(None,
                       feature_list=np.asarray([n[0] for n in rando_core_tuples]),
                       ids=np.asarray([n[1] for n in rando_core_tuples]))

    #now for each distractor class Sigma-add it to the core set
    dilu_level = 1
    for dilu_group in tqdm(np.array_split(rando_dilu_images, num_dilu_steps), desc='Dilution Groups'):
        dilu_only_tuples = []
        for q in dilu_group:
            dilu_only_tuples += image_path_tuple_list[q]
            compounding_dilu_tuples += image_path_tuple_list[q]

        dilued_tuples = rando_core_tuples + compounding_dilu_tuples

        # print(np.random.choice(dilu_group, num_add_queries))
        rando_queries_from_dilu += list(np.random.choice(dilu_group, num_add_queries))

        index_retrain = Index(gpu=True)
        # index_retrain = Index()
        # index_dilu = Index()

        s = time.time()
        index_retrain.train_index(None, training_features=np.asarray([n[0] for n in dilued_tuples]))
        e = time.time()
        tqdm.write(f'Training took {e - s}...')

        dilu_features = np.asarray([n[0] for n in dilued_tuples])
        dilu_ids = np.asarray([n[1] for n in dilued_tuples])

        #need to get ids
        s = time.time()
        index_retrain.add_to_index(None,
                           feature_list=dilu_features,
                           ids=dilu_ids)
        index_dilu.add_to_index(None,
                           feature_list=np.asarray([n[0] for n in dilu_only_tuples]),
                           ids=np.asarray([n[1] for n in dilu_only_tuples]))
        e = time.time()
        tqdm.write(f'Adding took {e - s}...')

        #need to get query features, need to query IMAGES not features, for loop?
        dilu_level_g_t_changes = 0
        query_precision_list_retrain = []
        query_precision_list_dilu = []
        gt_index = build_gt_index(dilu_features, dilu_ids)
        for query_image in tqdm(rando_query_images, desc='Core Queries'):
            query_tuples = image_path_tuple_list[query_image]

            query_feature_list = np.asarray([n[0] for n in query_tuples])
            query_id_list = np.asarray([n[1] for n in query_tuples])

            s = time.time()
            dists, ids_core = index_retrain.query_index(None, query_feature_list=query_feature_list, recall=feature_recall)
            voted_images_retrain = feature_to_image_voting(id_to_path, ids_core, recall=recall)


            dists, ids_dilu = index_dilu.query_index(None, query_feature_list=query_feature_list, recall=feature_recall)
            voted_images_dilu = feature_to_image_voting(id_to_path, ids_dilu, recall=recall)
            e = time.time()
            # tqdm.write(f'Querying and voting took {e - s}...')

            #compute g_t
            ids_g_t = query_gt(gt_index,
                               query_feature_list,
                               feature_recall)
            voted_images_g_t = feature_to_image_voting(id_to_path, ids_g_t, recall=recall)

            voted_paths_dilu = [n[0] for n in voted_images_dilu]
            voted_paths_retrain = [n[0] for n in voted_images_retrain]
            voted_paths_g_t = [n[0] for n in voted_images_g_t]

            if mAP:
                query_precision_list_retrain.append(mAP_metric(voted_paths_retrain, voted_paths_g_t, recall))
                query_precision_list_dilu.append(mAP_metric(voted_paths_dilu, voted_paths_g_t, recall))
            elif feature_level_matching:
                query_precision_list_retrain.append(mean(read_fvecs.compare_results_gt(ids_core, ids_g_t, recall=feature_recall)))
                query_precision_list_dilu.append(mean(read_fvecs.compare_results_gt(ids_dilu, ids_g_t, recall=feature_recall)))
            else:
                #get single precision score for retrained
                query_precision_list_retrain.append(len(list(set(voted_paths_retrain).intersection(voted_paths_g_t))) / recall)
                #get single precision score for dilued
                query_precision_list_dilu.append(len(list(set(voted_paths_dilu).intersection(voted_paths_g_t))) / recall)

            #each query image needs to remember its GT
            if rando_query_images_gt[query_image] == []:
                rando_query_images_gt[query_image] = voted_paths_g_t

            g_t_diff = len(rando_query_images_gt[query_image]) - len(list(set(rando_query_images_gt[query_image]).intersection(voted_paths_g_t)))
            if g_t_diff > 0:
                # tqdm.write(f'{query_image} ground truth has changed by {g_t_diff} items!')
                dilu_level_g_t_changes += g_t_diff
            rando_query_images_gt[query_image] = voted_paths_g_t

        results_json['trained_core_queries'][str(dilu_level)] += query_precision_list_retrain
        results_json['dilued_core_queries'][str(dilu_level)] += query_precision_list_dilu
        results_json['g_t_deltas'][str(dilu_level)].append(dilu_level_g_t_changes)

        query_precision_list_retrain = []
        query_precision_list_dilu = []
        # print(rando_queries_from_dilu)
        if dilu_set_queries:
            for query_image in tqdm(rando_queries_from_dilu, desc='Dilu Queries'):
                query_tuples = image_path_tuple_list[query_image]

                query_feature_list = np.asarray([n[0] for n in query_tuples])
                query_id_list = np.asarray([n[1] for n in query_tuples])

                s = time.time()
                dists, ids_core = index_retrain.query_index(None, query_feature_list=query_feature_list, recall=feature_recall)
                voted_images_retrain = feature_to_image_voting(id_to_path, ids_core, recall=recall)

                dists, ids_dilu = index_dilu.query_index(None, query_feature_list=query_feature_list, recall=feature_recall)
                voted_images_dilu = feature_to_image_voting(id_to_path, ids_dilu, recall=recall)
                e = time.time()
                # tqdm.write(f'Querying and voting took {e - s}...')

                #compute g_t
                ids_g_t = query_gt(gt_index,
                                   query_feature_list,
                                   feature_recall)
                voted_images_g_t = feature_to_image_voting(id_to_path, ids_g_t, recall=recall)

                voted_paths_dilu = [n[0] for n in voted_images_dilu]
                voted_paths_retrain = [n[0] for n in voted_images_retrain]
                voted_paths_g_t = [n[0] for n in voted_images_g_t]

            if mAP:
                query_precision_list_retrain.append(mAP_metric(voted_paths_retrain, voted_paths_g_t, recall))
                query_precision_list_dilu.append(mAP_metric(voted_paths_dilu, voted_paths_g_t, recall))
            elif feature_level_matching:
                query_precision_list_retrain.append(mean(read_fvecs.compare_results_gt(ids_core, ids_g_t, recall=feature_recall)))
                query_precision_list_dilu.append(mean(read_fvecs.compare_results_gt(ids_dilu, ids_g_t, recall=feature_recall)))
            else:
                #get single precision score for retrained
                query_precision_list_retrain.append(len(list(set(voted_paths_retrain).intersection(voted_paths_g_t))) / recall)
                #get single precision score for dilued
                query_precision_list_dilu.append(len(list(set(voted_paths_dilu).intersection(voted_paths_g_t))) / recall)

            results_json['trained_dilu_queries'][str(dilu_level)] += query_precision_list_retrain
            results_json['dilued_dilu_queries'][str(dilu_level)] += query_precision_list_dilu

        dilu_level += 1

print('Finished')
with open(f'./surf_{dataset}_q{num_queries}_r{recall}_dq{num_add_queries}_ds{num_dilu_steps}_t{trials}_results.json', 'w+') as f:
    json.dump(results_json, f)
