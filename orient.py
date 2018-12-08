#!/usr/bin/env python
import sys
import heapq
import math
import pickle
import random
import numpy as np

#### Model Parameter

### For nearest k
NEAREST_K = 10

### For adaboost
NUM_ADABOOST_CLASSIFIER = 16

### For forest
# the number of tree in forest
NUM_OF_TREE = 700
DATA_PER_TREE = 60

THRESHOLD = 130

# Avoid overfitting
MAX_TREE_DEPTH = 44

# How many time we try to sample idx to split the data
SPLIT_SAMPLING = 100


# General function
def general_extract_image_data(img_info):
    data = img_info.split(' ')
    return {
        'filename': data[0],
        'label': data[1],
        'data': [int(x) for x in data[2:]]
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
        'data': [int(x) for x in data[2:]]
    }


def get_label_dist(dataset):
    pred_ret = {'0': 0, '90': 0, '180': 0, '270': 0}
    for row in dataset:
        pred_ret[row["label"]] += 1
    return (max(pred_ret, key=pred_ret.get), pred_ret)


def vector_diff(img_data1, img_data2):
    diff = 0
    for i in range(192):
        v_d = (img_data1["data"][i] - img_data2["data"][i])
        diff += v_d * v_d
    return diff


def nearest_classify(model_nearest, test_data):
    # find nearest K
    # we need priority queue to keep the best K cand in the queue

    cands = []
    for cand in model_nearest:
        diff = vector_diff(cand, test_data)
        heapq.heappush(cands, (-diff, cand["label"]))
        if len(cands) > NEAREST_K:
            removed = heapq.heappop(cands)

    # get the result
    pred_ret = {'0': 0, '90': 0, '180': 0, '270': 0}
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
                heapq.heappush(cands_map[best_label], (
                    freq_map[best_label],
                    i,
                    j,
                    best_label,
                ))
                if len(cands_map[best_label]) > NUM_ADABOOST_CLASSIFIER / 4:
                    heapq.heappop(cands_map[best_label])

    hypotheses = []
    # get the best classifier from cands
    for i in range(NUM_ADABOOST_CLASSIFIER / 4):
        for label_type in cands_map:
            (freq, i, j, label) = heapq.heappop(cands_map[label_type])
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

        # randomly sampling to get a good split
        for i in range(SPLIT_SAMPLING):
            idx = random.randint(0, 191)
            if len(try_idx) < 192:
                while idx in try_idx:
                    idx = random.randint(0, 191)

                result = get_split_by_idx_entropy(dataset, idx)
                if result[0] < best_result[0]:
                    best_result = result
                # we have tried this idx
            try_idx.append(idx)

        (entropy, best_idx, best_gt_label, best_lt_label) = best_result

        # append selected idx to set
        new_choosed_idx = choosed_idx[:]
        choosed_idx.append(best_idx)

        (lt_data, gt_data) = split_data_by_idx(
            dataset,
            best_idx,
        )
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
    data_interval_per_tree = len(dataset) / NUM_OF_TREE

    for i in range(NUM_OF_TREE):
        tree_data = dataset[i * data_interval_per_tree:
                            i * data_interval_per_tree + DATA_PER_TREE]
        if i * data_interval_per_tree + DATA_PER_TREE > len(dataset):
            tree_data = tree_data + dataset[0:(
                i * data_interval_per_tree + DATA_PER_TREE) % len(dataset)]
        forest.append(build_tree(tree_data, []))

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
    elif model == 'best':
        model_best = train_adaboost_model(train_data)
        with open(output_file, 'w') as file:
            pickle.dump(model_best, file)


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

    elif model == 'best':
        model_best = {}
        with open(model_file, 'r') as file:
            model_best = pickle.load(file)

        for row in test_data:
            result.append(
                [row["filename"],
                 adaboost_classify(model_best, row)])

    # for testing
    # print(model)
    # Evaluate performance
    data_len = len(test_data)

    correct = 0
    for i in range(data_len):
        if test_data[i]["label"] == result[i][1]:
            correct += 1
        # else:
        #     print('Incorrect', test_data[i]["filename"])

    print("Accuracy", 1.0 * correct / data_len)

    # for row in result:
    #     print(" ".join(row))
    #
    # export results to file
    export_result_to_file(result)
    return


task_type = sys.argv[1]

if task_type == "train":
    train(sys.argv[2], sys.argv[3], sys.argv[4])
elif task_type == "test":
    test(sys.argv[2], sys.argv[3], sys.argv[4])

## Test script
##
##
##

#
# Test nearest
# train("/Users/cyli/code/cli3-a4/train-data-s.txt",
#       "/Users/cyli/code/cli3-a4/nearest_model.txt", "nearest")

# test("/Users/cyli/code/cli3-a4/test-data.txt",
#      "/Users/cyli/code/cli3-a4/nearest_model.txt", "nearest")

#
#
#
# Test adaboost

# train("/Users/cyli/code/cli3-a4/train-data.txt",
#       "/Users/cyli/code/cli3-a4/adaboost_model.txt", "adaboost")

# test("/Users/cyli/code/cli3-a4/test-data.txt",
#      "/Users/cyli/code/cli3-a4/adaboost_model.txt", "adaboost")

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