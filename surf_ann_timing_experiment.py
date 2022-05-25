import json, collections, os, time
import read_fvecs, find_ground_truth

import numpy as np

from tqdm import tqdm
from itertools import chain

from index_class import Index
from feature_extractor import feature_detection_and_description

def prepare_json(dilu_steps):
    results_json = {}
    results_json['IVF'] = {}
    results_json['flat'] = {}

    for i in range(1, dilu_steps + 1):
        results_json['IVF'][str(i)] = {}
        results_json['flat'][str(i)] = {}
        results_json['IVF'][str(i)]['train'] = []
        results_json['flat'][str(i)]['train'] = [0]
        results_json['IVF'][str(i)]['add'] = []
        results_json['flat'][str(i)]['add'] = []
        results_json['IVF'][str(i)]['query'] = []
        results_json['flat'][str(i)]['query'] = []

    return results_json

def feature_to_image_voting(id_to_path, returned_feature_ids, recall=10):
    s = time.time()
    voted_images = collections.Counter()

    for feature_result_ids in returned_feature_ids:
        for result_id in feature_result_ids:
            voted_images[id_to_path[result_id]] += 1

    e = time.time()
    tqdm.write(f'Voting took {e - s}')
    return voted_images.most_common(recall)

def feature_extraction(img_path):
    keyps, feats, det_t, dsc_t = feature_detection_and_description(img_path)
    return feats

def dilute(core, destractor, dilu_amount):
    return np.concatenate([core, destractor[:dilu_amount]])

def compute_diluted_gt(feature_vectors_in_index, ids, query_feature_vectors):
    s = time.time()
    i = find_ground_truth.build_flat_index(feature_vectors_in_index, ids, dim=64)
    e = time.time()

    results_json['flat'][str(dilu_level)]['add'].append(e - s)

    s = time.time()
    dists, ids = find_ground_truth.find_ground_truth(query_feature_vectors, i, recall=10)
    e = time.time()

    results_json['flat'][str(dilu_level)]['query'].append(e - s)

    return ids

dilu_steps = 5
recall = 25
n_queries = 50
dataset = 'apu'

load_from_images = True
images_root_path = '/home/wtheisen/Dropbox/4chan/downloads/Memes/Apu'

id_to_path = {}
feature_to_id = []

training_features = []
feature_list = []
query_features = []

results_json = prepare_json(dilu_steps)
tqdm.write(f'Loading dataset {dataset}')

img_features_dict = {}

if load_from_images:
    img_count = 0

    s = time.time()
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

        img_features_dict[image] = ided_feats

        id_to_path[img_count] = image
        img_count += 1
    e = time.time()
tqdm.write(f'Loading features took {e - s}')

for i in tqdm(range(0, 5), desc='Trials'):

    #select core set from the full feature list
    image_list = list(img_features_dict.keys())
    np.random.shuffle(image_list)
    rando_images = np.array_split(image_list, 2)

    rando_core_images = rando_images[0]
    rando_dilu_images = rando_images[1]

    rando_dilu_images = np.array_split(rando_dilu_images, dilu_steps)

    rando_query_images = np.random.choice(rando_core_images, n_queries)

    rando_core_tuples = []
    for i in rando_core_images:
            rando_core_tuples += img_features_dict[i]

    compounding_dilu_tuples = []

    #now for each distractor class Sigma-add it to the core set
    dilu_level = 1
    for dilu_set in tqdm(rando_dilu_images, desc='Dilutions', position=1):
        for dilu_image in dilu_set:
            compounding_dilu_tuples += img_features_dict[dilu_image]

        dilued_tuples = rando_core_tuples + compounding_dilu_tuples

        index_retrain = Index()
        index_dilu = Index()

        s = time.time()
        index_retrain.train_index(None, training_features=np.asarray([i[0] for i in dilued_tuples]))
        e = time.time()

        results_json['IVF'][str(dilu_level)]['train'].append(e - s)

        #need to get ids
        s = time.time()
        index_retrain.add_to_index(None,
                           feature_list=np.asarray([i[0] for i in dilued_tuples]),
                           ids=np.asarray([i[1] for i in dilued_tuples]))
        e = time.time()
        results_json['IVF'][str(dilu_level)]['add'].append(e - s)

        #need to get query features, need to query IMAGES not features, for loop?
        query_precision_list_retrain = []
        query_precision_list_dilu = []
        for query_image in tqdm(rando_query_images, desc='Queries', position=2):
            query_tuples = [img_features_dict[query_image] for q in rando_query_images]
            query_feature_list = np.asarray([i[0] for i in query_tuples[0]])
            query_id_list = np.asarray([i[1] for i in query_tuples[0]])
            # query_feature_list = np.asarray(list(chain(*query_tuples)))

            s = time.time()
            dists, ids_core = index_retrain.query_index(None, query_feature_list=query_feature_list, recall=recall)
            voted_images_retrain = feature_to_image_voting(id_to_path, ids_core, recall)
            e = time.time()

            results_json['IVF'][str(dilu_level)]['query'].append(e - s)

            # dists, ids_dilu = index_dilu.query_index(None, query_feature_list=query_feature_list, recall=recall)
            # voted_images_dilu = feature_to_image_voting(id_to_path, ids_dilu, recall)

            #compute g_t
            ids_g_t = compute_diluted_gt(np.asarray([i[0] for i in dilued_tuples]),
                                       np.asarray([i[1] for i in dilued_tuples]),
                                       np.asarray([i[0] for i in query_tuples[0]]))
            voted_images_g_t = feature_to_image_voting(id_to_path, ids_g_t, recall)

            #get single precision score for retrained
            # query_precision_list_retrain.append(len(list(set([i[0] for i in voted_images_retrain]).intersection([i[0] for i in voted_images_g_t]))) / 10)
            #get single precision score for dilued
            # query_precision_list_dilu.append(len(list(set([i[0] for i in voted_images_dilu]).intersection([i[0] for i in voted_images_g_t]))) / 10)

        # results_json['trained'][str(dilu_level)] += query_precision_list_retrain
        # results_json['dilued'][str(dilu_level)] += query_precision_list_dilu
        dilu_level += 1

tqdm.write('Finished')
with open('./timings.json', 'w+') as f:
    json.dump(results_json, f)
