import json
import matplotlib.pyplot as plt

from imgcat import imgcat

def box_plotter(file_path):
    with open(file_path) as f:
        results_json = json.load(f)

    data_core_queries = []
    data_dilu_queries = []

    gt_avg_change = []

    # for i in zip(list(results_json['trained_core_queries'].values()),
    #         list(results_json['dilued_core_queries'].values()),
    #         list(results_json['trained_dilu_queries'].values()),
    #         list(results_json['dilued_dilu_queries'].values()),
    #         list(results_json['g_t_deltas'].values())):
    for i in list(results_json['trained'].keys()):
            data_core_queries.append(results_json['trained'][i])
            data_dilu_queries.append(results_json['dilued'][i])

    #     if i[0] != []:
    #         data_core_queries.append(i[0])
    #         data_core_queries.append(i[1])

    #         data_dilu_queries.append(i[2])
    #         data_dilu_queries.append(i[3])

    #         gt_avg_change.append(i[4][0] / recall)

    labels = []
    positions = []
    init = 1.0

    trained_x = []
    dilued_x = []

    # for i in range(0, len(data_core_queries), 2):
    for i in range(0, len(data_core_queries)):
        positions.append(init)
        positions.append(init + 0.6)

        trained_x.append(init)
        dilued_x.append(init + 0.6)

        init += 1.5

        labels.append(str(i * 100) + '%_T')
        labels.append(str(i * 100) + '%_D')

    # positions = positions[:-1]
    # labels = labels[:-1]


    # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig, ax1 = plt.subplots()

    ax_right = ax1.twinx()

    print(len(data_core_queries), len(data_dilu_queries), len(positions), len(labels))

    test = [v for p in zip(data_core_queries, data_dilu_queries) for v in p]

    ax1.boxplot(test, 0, '',
        positions=positions,
        labels=labels)

    # ax2.boxplot(data_dilu_queries, 0, '',
    #     positions=positions,
    #     labels=labels)

    ax1.set_xlabel('Dilution level')
    ax1.set_ylabel('Retrieval Accuracy')
    # ax2.set_ylabel('mAP (Image, Dilu)')

    # ax_right.plot(trained_x, gt_avg_change, label='Average GT delta')
    # ax_right.set_ylabel('Average GT change')

    # ax2.xaxis.set_tick_params(rotation=-45)
    ax1.xaxis.set_tick_params(rotation=-45)
    plt.legend()
    plt.title('SIFT-10k Retrieval Accuracy Decay (10 time steps)')

    imgcat(fig)

if __name__ == '__main__':
    box_plotter('./sift_1m_rewrite_results_step_retrain.json')
