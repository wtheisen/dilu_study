import json
import random
import read_fvecs, find_ground_truth

import numpy as np

from collections import defaultdict
from index_class import Index
from feature_extractor import feature_detection_and_description


def feature_extraction(img_path):
    keyps, feats, det_t, dsc_t = feature_detection_and_description(img_path)
    return feats


def dilute(core, destractor, dilu_amount):
    return np.concatenate([core, destractor[:dilu_amount]])


def compute_diluted_gt(feature_vectors_in_index, ids, query_feature_vectors, recall):
    print('computing new ground truth')
    i = find_ground_truth.build_flat_index(feature_vectors_in_index, ids)
    dists, ids = find_ground_truth.find_ground_truth(np.asarray(query_feature_vectors), i, recall=recall)
    return ids


# with open('./sift_results_template.json') as f:
#     results_json = json.load(f)

results_json = {}
results_json['trained'] = defaultdict(list)
results_json['dilued'] = defaultdict(list)

trials = 10
dilu_steps = 10

retrain_step_size = 2

queries = 10000
recall = 50

dataset = 'sift1m'
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

for i in range(0, trials):
    # print('computing core, trial:', i)
    # if compute_core:
    #     core_tuples = list(read_fvecs.compute_small_feature_set(feature_list, g_t).items())
    #     np.random.shuffle(core_tuples)

    # print('computing distractors, trial:', i)
    # if compute_distractors:
    #     print('Splitting core tuple')
    #     ids, features = map(list, zip(*core_tuples))
    #     # destractor_tuples = list(read_fvecs.compute_destractor_set([i[0] for i in core_tuples], feature_list).items())
    #     destractor_tuples = list(read_fvecs.compute_destractor_set(ids, feature_list).items())
    #     print('Shuffling destractor features')
    #     np.random.shuffle(destractor_tuples)

    # full_tuples = np.concatenate([core_tuples, destractor_tuples])

    full_tuples = []
    for i in range(0, len(feature_list)):
        full_tuples.append((i, feature_list[i]))

    print('shuffling')
    np.random.shuffle(full_tuples)

    # np.random.shuffle(feature_list)
    print('splitting data')
    split_data = np.array_split(full_tuples, dilu_steps)
    # split_core_tuples = np.array_split(core_tuples, 4)
    # split_destractor_tuples = np.array_split(destractor_tuples, 4)

    # rando_core = np.concatenate([split_core_tuples[0], split_destractor_tuples[0]])
    # rando_dilu = np.concatenate([split_core_tuples[1], split_destractor_tuples[1], split_destractor_tuples[2], split_destractor_tuples[3]])
    rando_core = split_data[0]
    query_features = random.sample([i[1] for i in rando_core], queries)
    # rando_dilu = np.concatenate([i for i in split_data[1:]])
    # np.random.shuffle(rando_dilu)
    # dilu_sets = np.array_split(full_tuples, 9)

    index_core = Index(gpu=True, feature_type='SIFT')
    index_dilu = Index(gpu=True, feature_type='SIFT')

    index_core.train_index(None, training_features=np.asarray([i[1] for i in rando_core]))
    drop_training_list = []

    for dilu_lvl in range(0, len(split_data)):

        dilu_tuples = split_data[dilu_lvl]
        dilu_ids, dilu_features = map(list, zip(*dilu_tuples))

        current_tuple_set = np.concatenate([i for i in split_data[0:dilu_lvl+1]])
        t_ids, t_features = map(list, zip(*current_tuple_set))

        index_dilu.train_index(None, training_features=np.asarray(t_features))
        if dilu_lvl % retrain_step_size == 0 and dilu_lvl != 0:
            drop_training_list.append('Drop training, dilu_lvl: ' + str(dilu_lvl))
            current_tuple_set = np.concatenate([i for i in split_data[0:dilu_lvl+1:retrain_step_size]])
            t_ids, t_features = map(list, zip(*current_tuple_set))

            index_core.train_index(None, training_features=np.asarray(t_features))

        # print(dilu_features[:10])
        # print(dilu_ids[:10])
        # exit()

        index_core.add_to_index(None,
                           feature_list=np.asarray(dilu_features),
                           ids=np.asarray(dilu_ids))
        index_dilu.add_to_index(None,
                           feature_list=np.asarray(dilu_features),
                           ids=np.asarray(dilu_ids))

        dists, ids_core = index_core.query_index(None, query_feature_list=np.asarray(query_features), recall=recall)
        dists, ids_dilu = index_dilu.query_index(None, query_feature_list=np.asarray(query_features), recall=recall)

        print('Number of feature vectors trained on:', len(rando_core))
        print('Dilution level:', dilu_lvl)
        print('Total feature vectors', len(dilu_tuples))

        g_t = compute_diluted_gt(np.asarray(dilu_features),
                               np.asarray(dilu_ids),
                               query_features,
                               recall)

        print('Re-trained Index:')
        query_acc_list = read_fvecs.compare_results_gt(ids_dilu, g_t, recall=recall)
        results_json['trained'][str(dilu_lvl)] += query_acc_list
        print('Diluted Index:')
        query_acc_list = read_fvecs.compare_results_gt(ids_core, g_t, recall=recall)
        results_json['dilued'][str(dilu_lvl)] += query_acc_list

print('Finished')
print(drop_training_list)
# print(results_json)

with open('./sift_1m_rewrite_results_step_retrain.json', 'w+') as f:
    json.dump(results_json, f)
