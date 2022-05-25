import json
import read_fvecs, find_ground_truth

import numpy as np

from index_class import Index
from feature_extractor import feature_detection_and_description

def feature_extraction(img_path):
    keyps, feats, det_t, dsc_t = feature_detection_and_description(img_path)
    return feats

def dilute(core, destractor, dilu_amount):
    return np.concatenate([core, destractor[:dilu_amount]])

def compute_diluted_gt(feature_vectors_in_index, ids, query_feature_vectors):
    print('computing new ground truth')
    i = find_ground_truth.build_flat_index(feature_vectors_in_index, ids)
    dists, ids = find_ground_truth.find_ground_truth(query_feature_vectors, i, recall=recall)
    return ids

with open('./sift_results_template.json') as f:
    results_json = json.load(f)

recall = 10
dataset = 'sift10k'
compute_core = True
compute_distractors = True

load_from_images = False
image_list_path = ''

id_to_path = {}
feature_to_id = []

training_features = []
feature_list = []
query_features = []

print('Loading dataset', dataset)

if load_from_images:
    img_count = 0
    with open(image_list_path) as f:
        image_path_list = f.readlines().rstrip().lstrip()

    for image in image_path_list:
        feats = feature_extraction(image)
        for f in feats:
            feature_to_id.append((f, img_count))
        training_features.append(f)
        feature_list.append(f)
        id_to_path[img_count] = image
        img_count += 1

elif dataset == 'sift10k':
    g_t = read_fvecs.ivecs_read('/media/wtheisen/scratch2/siftsmall/siftsmall_groundtruth.ivecs')
    training_features = read_fvecs.load_sift_features('/media/wtheisen/scratch2/siftsmall/siftsmall_learn.fvecs')
    feature_list = read_fvecs.load_sift_features('/media/wtheisen/scratch2/siftsmall/siftsmall_base.fvecs')
    query_features = read_fvecs.load_sift_features('/media/wtheisen/scratch2/siftsmall/siftsmall_query.fvecs')

elif dataset == 'sift1m':
    g_t = read_fvecs.ivecs_read('/media/wtheisen/scratch2/sift1m/sift_groundtruth.ivecs')
    training_features = read_fvecs.load_sift_features('/media/wtheisen/scratch2/sift1m/sift_learn.fvecs')
    feature_list = read_fvecs.load_sift_features('/media/wtheisen/scratch2/sift1m/sift_base.fvecs')
    query_features = read_fvecs.load_sift_features('/media/wtheisen/scratch2/sift1m/sift_query.fvecs')

for i in range(0, 10):
    print('computing core, trial:', i)
    if compute_core:
        core_tuples = list(read_fvecs.compute_small_feature_set(feature_list, g_t).items())
        np.random.shuffle(core_tuples)

    print('computing distractors, trial:', i)
    if compute_distractors:
        print('Splitting core tuple')
        ids, features = map(list, zip(*core_tuples))
        # destractor_tuples = list(read_fvecs.compute_destractor_set([i[0] for i in core_tuples], feature_list).items())
        destractor_tuples = list(read_fvecs.compute_destractor_set(ids, feature_list).items())
        print('Shuffling destractor features')
        np.random.shuffle(destractor_tuples)

    full_tuples = np.concatenate([core_tuples, destractor_tuples])
    print('shuffling')
    np.random.shuffle(full_tuples)

    # np.random.shuffle(feature_list)
    print('splitting data')
    split_data = np.array_split(full_tuples, 10)
    # split_core_tuples = np.array_split(core_tuples, 4)
    # split_destractor_tuples = np.array_split(destractor_tuples, 4)

    # rando_core = np.concatenate([split_core_tuples[0], split_destractor_tuples[0]])
    # rando_dilu = np.concatenate([split_core_tuples[1], split_destractor_tuples[1], split_destractor_tuples[2], split_destractor_tuples[3]])
    rando_core = split_data[0]
    rando_dilu = np.concatenate([i for i in split_data[1:]])
    # np.random.shuffle(rando_dilu)

    for dilu in range(0, 9500, 500):
        index_core = Index(gpu=True, feature_type='SIFT')
        index_dilu = Index(gpu=True, feature_type='SIFT')

        dilu_tuples = dilute(rando_core, rando_dilu, dilu)
        dilu_ids, dilu_features = map(list, zip(*dilu_tuples))

        index_core.train_index(None, training_features=np.asarray([i[1] for i in rando_core]))
        index_dilu.train_index(None, training_features=np.asarray(dilu_features))

        index_core.add_to_index(None,
                           feature_list=np.asarray(dilu_features),
                           ids=np.asarray(dilu_ids))
        index_dilu.add_to_index(None,
                           feature_list=np.asarray(dilu_features),
                           ids=np.asarray(dilu_ids))

        dists, ids_core = index_core.query_index(None, query_feature_list=query_features, recall=recall)
        dists, ids_dilu = index_dilu.query_index(None, query_feature_list=query_features, recall=recall)

        print('Number of feature vectors trained on:', len(rando_core))
        print('Dilution level:', dilu)
        print('Total feature vectors', len(dilu_tuples))

        g_t = compute_diluted_gt(np.asarray(dilu_features),
                                   np.asarray(dilu_ids),
                                   query_features)
        print('Re-trained Index:')
        query_acc_list = read_fvecs.compare_results_gt(ids_dilu, g_t, recall=recall)
        results_json['trained'][str(dilu)] += query_acc_list
        print('Diluted Index:')
        query_acc_list = read_fvecs.compare_results_gt(ids_core, g_t, recall=recall)
        results_json['dilued'][str(dilu)] += query_acc_list

print('Finished')
print(results_json)

with open('./sift_10k_mega_dilu_r10_results', 'w+') as f:
    json.dump(results_json, f)
