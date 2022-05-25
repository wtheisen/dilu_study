import numpy as np
from tqdm import tqdm

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def load_sift_features(feature_file):
    xt = fvecs_read(feature_file)
    # print('Loaded features:', xt)
    # print('Shape:', xt.shape)

    return xt

def compute_small_feature_set(full_features, gt):
    gt_vec_ids = set()
    for q in gt:
        gt_vec_ids |= set(q)

    small_feature_dict = {}

    for i in range(0, len(full_features)):
        if i in gt_vec_ids:
            small_feature_dict[i] = full_features[i]


    # print('num core features:', len(small_feature_dict.keys()))
    # print('num tot features:', len(full_features))
    return small_feature_dict

def compute_destractor_set(non_destractors, full_features):
    destractor_features_dict = {}
    non_destractors = set(non_destractors)

    for i in tqdm(range(0, len(full_features))):
        if i not in non_destractors:
            destractor_features_dict[i] = full_features[i]

    return destractor_features_dict

def compare_results_gt(results, gt, recall=100):
    count = 1
    query_percent_dict = {}

    for q_r, g_t in zip(results, gt):
        # print('q_r', q_r)
        # print('g_t', g_t)
        cor_count = 0
        for r in q_r:
            if r in g_t[:recall]:
                cor_count += 1
        query_percent_dict[str(count)] = float(cor_count / recall)

        count += 1

    # print(query_percent_dict)
    q_accs = list(query_percent_dict.values())
    return q_accs

    print('min:', min(q_accs))
    q1 = np.quantile(q_accs, 0.25)
    print('q1:', q1)
    q2 = np.quantile(q_accs, 0.50)
    print('q2:', q2)
    q3 = np.quantile(q_accs, 0.75)
    print('q3:', q3)
    print('max:', max(q_accs))
    return min(q_accs), q1, q2, q3, max(q_accs)
