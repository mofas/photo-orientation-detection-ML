import sys
import heapq
import math
import pickle
import random
import numpy as np


# General function
def general_extract_image_data(img_info):
    data = img_info.split(' ')
    return {
        'filename': data[0],
        'label': data[1],
        'data': [int(x) for x in data[1:]]
    }


# for nearest model
def extract_nearest_model_data(img_info):
    data = img_info.split(' ')
    return {'label': data[0], 'data': [int(x) for x in data[1:]]}


def extract_adaboost_train_data(img_info, initial_weight):
    data = img_info.split(' ')
    return {
        'weight': initial_weight,
        'label': data[1],
        'data': [int(x) for x in data[1:]]
    }


def get_label_dist(dataset):
    pred_ret = {'0': 0, '90': 0, '180': 0, '270': 0}
    for row in dataset:
        pred_ret[row["label"]] += 1
    return (max(pred_ret, key=pred_ret.get), pred_ret)


# img1 = "train/6179019779.jpg 180 142 135 134 148 141 139 146 139 137 134 127 122 126 121 119 87 81 87 107 124 155 112 132 167 135 128 125 115 108 102 104 97 94 111 108 110 110 124 149 123 141 172 122 140 173 120 139 173 126 120 116 108 113 126 103 116 141 127 143 173 128 148 181 129 148 181 129 148 183 127 146 184 143 155 178 142 160 191 139 157 188 130 147 177 110 131 163 126 147 180 143 162 196 143 162 197 91 114 149 93 115 150 86 108 141 70 92 125 61 85 119 66 93 128 106 132 167 137 161 197 83 110 148 92 117 154 98 123 158 92 118 154 94 117 148 97 124 155 88 119 160 91 122 164 201 212 227 196 209 225 199 213 228 204 216 229 207 219 232 203 218 233 176 194 215 173 190 212 197 210 231 177 197 225 172 194 225 189 206 231 200 214 235 198 214 237 213 226 244 218 229 245"
# img2 = "train/6143621243.jpg 0 151 178 209 131 169 210 147 176 211 172 184 211 125 162 208 100 149 209 99 145 207 98 142 204 180 203 227 184 208 231 179 206 230 166 200 232 150 195 235 115 173 225 99 162 222 89 152 219 223 232 234 220 231 236 217 237 247 190 220 236 117 146 170 159 198 228 143 188 226 123 175 220 238 244 240 246 255 255 233 235 236 136 130 135 119 116 136 223 241 253 213 230 239 205 221 232 202 181 149 216 193 171 218 136 122 207 109 93 206 148 123 222 215 174 211 199 168 229 208 168 88 79 76 130 84 88 167 106 108 111 80 75 84 77 59 96 77 86 86 70 78 63 57 42 33 39 17 40 42 18 42 45 22 45 48 22 40 42 24 46 44 41 33 38 22 15 27 7 19 29 8 24 33 8 28 35 8 26 35 9 28 35 8 23 32 4 28 36 13 27 34 18"

# img_data1 = extract_image_data(img1)
# img_data2 = extract_image_data(img2)
model_best = []

#### Hyperparameter

### For nearest k
NEAREST_K = 50

### For adaboost
NUM_ADABOOST_CLASSIFIER = 500

### For forest
# the number of tree in forest
NUM_OF_TREE = 400
THRESHOLD = 180

# Avoid overfitting
MAX_TREE_DEPTH = 16

# How many time we try to sample idx to split the data
SPLIT_SAMPLING = 10


def vector_diff(img_data1, img_data2):
    diff = 0
    for i in range(192):
        diff += (img_data1["data"][i] - img_data2["data"][i])**2
    return diff


def nearest_classify(model_nearest, test_data):
    # find nearest K
    # we need priority queue to keep the best K cand in the queue

    cands = []
    pred_ret = {'0': 0, '90': 0, '180': 0, '270': 0}
    for cand in model_nearest:
        diff = vector_diff(cand, test_data)
        heapq.heappush(cands, (-diff, cand["label"]))
        if len(cands) > NEAREST_K:
            removed = heapq.heappop(cands)
            # print("removed", removed)

    # get the result
    while len(cands):
        (score, pred) = heapq.heappop(cands)
        pred_ret[pred] += 1
        # print("in", (score, pred))

    # print(pred_ret, max(pred_ret, key=pred_ret.get))
    return max(pred_ret, key=pred_ret.get)


def get_hypotheses(dataset):
    cands_map = {'0': [], '90': [], '180': [], '270': []}

    # Building the frequency mapping
    for i in range(192):
        for j in range(192):
            if i != j:
                freq_map = {'0': 0, '90': 0, '180': 0, '270': 0}
                for row in dataset:
                    if row["data"][i] > row["data"][j]:
                        freq_map[row["label"]] += 1
                # print(freq_map)
                best_label = max(freq_map, key=freq_map.get)
                heapq.heappush(cands_map[best_label],
                               (-freq_map[best_label], i, j, best_label,
                                freq_map[best_label]))

    hypotheses = []
    # get the best classifier from cands
    for i in range(NUM_ADABOOST_CLASSIFIER / 4):
        for label_type in cands_map:
            (score, i, j, label, freq) = heapq.heappop(cands_map[label_type])
            hypotheses.append((i, j, label, freq))
    return hypotheses


def normalize_weight(dataset):
    weight_sum = 0
    for data in dataset:
        weight_sum += data["weight"]

    for data in dataset:
        data["weight"] = data["weight"] / weight_sum


def train_adaboost_model(train_data):
    dataset = []
    for row in train_data:
        dataset.append(extract_adaboost_train_data(row, 1.0 / len(train_data)))

    hypotheses = get_hypotheses(dataset)

    model = []
    for h in hypotheses:
        (i, j, label, _) = h

        error = 0
        # calculate the error of incorrect classify
        for data in dataset:
            if data["data"][i] > data["data"][j] and data["label"] != label:
                error += data["weight"]

        # adjust the weight of data which we predict correctly
        for data in dataset:
            if data["data"][i] > data["data"][j] and data["label"] == label:
                data["weight"] = data["weight"] * error / (1 - error)

        # normalize weight
        normalize_weight(dataset)

        h_weight = math.log((1 - error) / error)
        model.append((i, j, label, h_weight))

    # the model should be something like
    # [(i, j, label, weight) ... ]
    return model


def adaboost_classify(model_adaboost, test_data):
    pred_ret = {'0': 0, '90': 0, '180': 0, '270': 0}

    for hypothesis in model_adaboost:
        (i, j, label, weight) = hypothesis
        if i != j and test_data["data"][i] > test_data["data"][j]:
            pred_ret[label] += weight

    return max(pred_ret, key=pred_ret.get)


def get_split_by_idx_entropy(dataset, idx):

    total = len(dataset)
    gt_pred_ret = {'0': 0, '90': 0, '180': 0, '270': 0}
    lt_pred_ret = {'0': 0, '90': 0, '180': 0, '270': 0}
    for d in dataset:
        if d["data"][idx] > THRESHOLD:
            gt_pred_ret[d["label"]] += 1
        else:
            lt_pred_ret[d["label"]] += 1
    best_gt_label = max(gt_pred_ret, key=gt_pred_ret.get)
    best_lt_label = max(lt_pred_ret, key=lt_pred_ret.get)
    data_in_gt = sum(gt_pred_ret.values())
    data_in_lt = sum(lt_pred_ret.values())
    # print("====")
    # print(gt_pred_ret, best_gt_label)
    # print(lt_pred_ret, best_lt_label)

    # This idx is not useful if we get both same label on both direction
    if best_gt_label == best_lt_label:
        return (1, idx, best_gt_label, best_lt_label)

    # This idx is not useful if all the data are in one side
    if data_in_gt == 0 or data_in_lt == 0:
        return (1, idx, best_gt_label, best_lt_label)

    gt_n = (data_in_gt - gt_pred_ret[best_gt_label]) * 1.0 / data_in_gt
    lt_n = (data_in_lt - lt_pred_ret[best_lt_label]) * 1.0 / data_in_lt

    # perfect two side split
    if gt_n == 0 and lt_n == 0:
        entropy = 0
    # perfect one side split
    elif gt_n == 0:
        entropy = (data_in_lt * 1.0 / total) * -lt_n * math.log(lt_n)
    elif lt_n == 0:
        entropy = (data_in_gt * 1.0 / total) * -lt_n * math.log(gt_n)
    else:
        entropy = (data_in_gt * 1.0 / total) * -gt_n * math.log(gt_n) + (
            data_in_lt * 1.0 / total) * -lt_n * math.log(lt_n)
    return (entropy, idx, best_gt_label, best_lt_label)


def split_data_by_idx(dataset, idx):
    gt_data = []
    lt_data = []

    for d in dataset:
        if d["data"][idx] > THRESHOLD:
            gt_data.append(d)
        else:
            lt_data.append(d)

    return (lt_data, gt_data)


def build_tree(dataset, choosed_idx):
    data_len = len(dataset)
    (best_label, label_dist) = get_label_dist(dataset)

    # if tree is too deep, then we stop
    # if all data have same labels, we also stop
    # return best_label
    # Otherwise, we try to split dataset again
    if len(choosed_idx) < MAX_TREE_DEPTH and label_dist[best_label] != data_len:

        # random choose a idx which is not choosen before
        try_idx = choosed_idx[:]

        # the best result we find yet
        best_result = (2, 0, '0', '0')

        # randomly sampling try to get a good split
        for i in range(SPLIT_SAMPLING):
            idx = random.randint(0, 192)
            if len(try_idx) < 192:
                while idx in try_idx:
                    idx = random.randint(0, 192)
            result = get_split_by_idx_entropy(dataset, idx)

            if result[0] < best_result[0]:
                best_result = result
            # we have tried this idx
            try_idx.append(idx)

        (entropy, best_idx, best_gt_label, best_lt_label) = best_result

        # append selected idx to set
        choosed_idx.append(best_idx)

        (lt_data, gt_data) = split_data_by_idx(dataset, best_idx)
        return {
            "idx": best_idx,
            "lt": build_tree(lt_data, choosed_idx),
            "gt": build_tree(gt_data, choosed_idx),
        }
    else:
        # we return the prediction based on the most freq label
        return best_label


def tree_classify(tree, data):
    if isinstance(tree, basestring):
        return tree
    if data["data"][tree["idx"]] > THRESHOLD:
        return tree_classify(tree["gt"], data)
    else:
        return tree_classify(tree["lt"], data)


def train_forest_model(train_data):
    dataset = []
    for row in train_data:
        dataset.append(general_extract_image_data(row))

    forest = []

    # split dataset
    data_per_tree = len(dataset) / NUM_OF_TREE

    for i in range(NUM_OF_TREE):
        forest.append(
            build_tree(dataset[i * data_per_tree:(i + 1) * data_per_tree], []))

    return forest


def forest_classify(forest, data):
    pred_ret = {'0': 0, '90': 0, '180': 0, '270': 0}
    for tree in forest:
        pred_ret[tree_classify(tree, data)] += 1

    return max(pred_ret, key=pred_ret.get)


def train(input_file, output_file, model):
    with open(input_file, 'r') as file:
        train_data = file.readlines()

    # in nearest, we just rmeove the filename from data
    if model == 'nearest':
        with open(output_file, 'w') as file:
            for row in train_data:
                file.write(" ".join(row.split(' ')[1:]))
    elif model == 'adaboost':
        model_adaboost = train_adaboost_model(train_data)
        with open(output_file, 'w') as file:
            pickle.dump(model_adaboost, file)
    elif model == 'forest':
        model_forest = train_forest_model(train_data)
        with open(output_file, 'w') as file:
            pickle.dump(model_forest, file)
    return


def export_result_to_file(result):
    with open('output.txt', 'w') as file:
        for row in result:
            file.write(" ".join(row) + "\n")


def test(test_file, model_file, model):
    raw_data = []
    test_data = []
    result = []

    with open(test_file, 'r') as file:
        raw_data = file.readlines()

    for row in raw_data:
        test_data.append(general_extract_image_data(row))

    if model == 'nearest':
        with open(model_file, 'r') as file:
            model_data = file.readlines()

        model_nearest = []
        for row in model_data:
            model_nearest.append(extract_nearest_model_data(row))

        for row in test_data:
            result.append(
                [row["filename"],
                 nearest_classify(model_nearest, row)])

    elif model == 'adaboost':
        model_adaboost = {}
        with open(model_file, 'r') as file:
            model_adaboost = pickle.load(file)

        for row in test_data:
            result.append(
                [row["filename"],
                 adaboost_classify(model_adaboost, row)])

    elif model == 'forest':
        model_forest = {}
        with open(model_file, 'r') as file:
            model_forest = pickle.load(file)

        for row in test_data:
            result.append(
                [row["filename"],
                 forest_classify(model_forest, row)])

    # for testing
    print(model)
    # Evaluate performance
    data_len = len(test_data)

    correct = 0
    for i in range(data_len):
        if test_data[i]["label"] == result[i][1]:
            correct += 1

    print("Accuracy", 1.0 * correct / data_len)

    # for row in result:
    #     print(" ".join(row))
    #
    # export results to file
    # export_result_to_file(result)
    return


# TODO interface!!
# task_type = sys.argv[1]

# if task_type == "train":
#     train(sys.argv[2], sys.argv[3], sys.argv[4])
# elif task_type == "test":
#     test(sys.argv[2], sys.argv[3], sys.argv[4])

## Test script
##
##
##

#
# Test nearest
# train("/Users/cyli/code/cli3-a4/train-data-s.txt",
#       "/Users/cyli/code/cli3-a4/nearest_model.txt", "nearest")

# test("/Users/cyli/code/cli3-a4/test-data-s.txt",
#      "/Users/cyli/code/cli3-a4/nearest_model.txt", "nearest")

#
#
#
# Test adaboost

train("/Users/cyli/code/cli3-a4/train-data.txt",
      "/Users/cyli/code/cli3-a4/adaboost_model.txt", "adaboost")

test("/Users/cyli/code/cli3-a4/test-data.txt",
     "/Users/cyli/code/cli3-a4/adaboost_model.txt", "adaboost")

#
#
#
# Test forest

# train("/Users/cyli/code/cli3-a4/train-data.txt",
#       "/Users/cyli/code/cli3-a4/forest_model.txt", "forest")

# test("/Users/cyli/code/cli3-a4/test-data.txt",
#      "/Users/cyli/code/cli3-a4/forest_model.txt", "forest")

#
#
#
#
# # Full data training model

# train("/Users/cyli/code/cli3-a4/train-data.txt",
#       "/Users/cyli/code/cli3-a4/nearest_model.txt", "nearest")

# train("/Users/cyli/code/cli3-a4/train-data.txt",
#       "/Users/cyli/code/cli3-a4/adaboost_model.txt", "adaboost")

# train("/Users/cyli/code/cli3-a4/train-data.txt",
#       "/Users/cyli/code/cli3-a4/forest_model.txt", "forest")

#