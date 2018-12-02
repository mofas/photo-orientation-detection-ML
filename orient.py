#!/usr/bin/env python
import sys
import heapq
import math
import pickle
import random
import numpy as np

# Nearest model:
# Nearest model is very simple, we just look through all the images,
# and I use "vector_diff" function to find the accumulated square vector difference between
# two images in all 192 pixels. The only parameter we need to decide is how many "closest"
# images (NEAREST_K) to choose to vote for the final result.
# I try K = 5, K = 10, k = 20, and k = 50
# K = 5  : 0.5979
# K = 10 : 0.6177
# K = 50 : 0.6260
# K = 100: 0.6333
#
# However, the performance of the nearest model decrease significantly if we don't have enough
# data. For example, if we only have 1000 training image, the accuracy is lower than 0.3,
# which is pretty close random guess.
#
# Another drawback of the nearest model is its classified time take too long,
# because we actually compared the test image to all the training data.
#
# In short, nearest k model performed well when you have lots of data. However,
# the extremely long classify time to make this algorithm impractical.
#
#
# #
# Adaboost model:
# In AdaBoost model, I iterate all the possible weak classifier candidates (192*192)
# and choose the best n classifier (NUM_ADABOOST_CLASSIFIER) to form the
# final AdaBoost classifier. In addition, I find the lots of candidates are very
# good at label "0" images. While only a few candidates can do good jobs at label "270".
#
# Therefore, I choose the best NUM_ADABOOST_CLASSIFIER/4 candidates for all orientations.
# I believe through this mechanism, I can find the "experts" for all orientations.
#
# The following is the accuracy for different NUM_ADABOOST_CLASSIFIER.
#
# NUM_ADABOOST_CLASSIFIER = 2   : 0.2646
# NUM_ADABOOST_CLASSIFIER = 5   : 0.659375
# NUM_ADABOOST_CLASSIFIER = 10  : 0.678125
# NUM_ADABOOST_CLASSIFIER = 16  : 0.6927
# NUM_ADABOOST_CLASSIFIER = 20  : 0.6927
# NUM_ADABOOST_CLASSIFIER = 28  : 0.6271
# NUM_ADABOOST_CLASSIFIER = 50  : 0.553125
# NUM_ADABOOST_CLASSIFIER = 500 : 0.575
#
# You can see here that increase the number of weak classifiers will not increase
# the performance, even decrease it.
# I think I can explain that if there are too many too weak classifiers can vote,
# then it will create lots of noise and decrease the performance.
#
# The classify time of AdaBoost is blazing fast because we just summary the
# result of all classifiers.
#
# Another benefit of AdaBoost is it works quite well even we don't have lots of data.
# For 1000 data, the performance of AdaBoost can over 0.4, which is pretty good
# compared with the nearest K.
#
# However, the training time of AdaBoost is a drawback, for training around 40k
# data, it will take around 400 sec in my computer to train, which is much slower
# than forest model.
#
# #
# Forest model:
# Forest model is quite complex to implement because lots of decisions to make
# and parameters to tune.
# In my implementation, I use the value of a pixel as features to split dataset.
# If the value of certain pixel higher than parameter THRESHOLD, then
# we put them into the left subtree, otherwise, in the right subtree.
#
# In addition, when we split the data, which index to use is also important.
# In theory, we can try all 192 indexes and choose the one giving us the lowest entropy.
# However, I find if I randomly choose some good indexes but not the best.
# I will get the better performance in the end!
#
# This is an amazing result. Therefore, I have a parameter called "SPLIT_SAMPLING"
# to decide how many samplings we did for finding a good index.
#
# Another experiment I try it how many data per tree. Notice, if the number of tree
# multiple by the number of data per tree larger than the total data we have,
# we will need to "reuse" some data. I do the several experiments and I find if
# a data is reused in more than 2 trees, the performance will decrease
# gradually.
#
# Other parameter is how many trees we should have and how depth each tree will be.
# It is very hard to tune. Nevertheless, both of them should depend on how many
# train data you have. From my personal experience, if we give a tree too many data,
# then it will grow very high(deep), which caused the overfitting. If we give too few
# data, then the tree is underfitting. Similarity, growing too many trees or too few
# trees will decrease performance.
#
# I try the several parameters, and I find the optimal max tree depth is around 44
# (depend on how many data you feed into the tree),
# and the optimal tree number is around 500 ~ 800. The optimal threshold for
# splitting data is around 130 ~ 170. The number of data feeds to each tree is
# around 60. The number of sampling split index is around 50 ~ 100.
#
# The forest model has several advantages. First of all, relatively faster training
# time compared to AdaBoost, which help developers to tune parameters easily.
# Second, the classified time is also very fast, in just several seconds.
# Third, forest model doing quite well even we don't have lots of data.
# For 1000 data, the accuracy of forest model can over 0.55 if we tune the
# parameter properly. Compared with 0.3 of nearest K, and 0.4 of AdaBoost,
# the performance is unbelievably good.
#
# However, the main disadventage of the forest model is that we need to
# tune the parameters by hand, which is a time consuming job. The worst of all,
# the performance landscape is quite rough, there is no easy way to run
# gradient descent algorithm to find the optimal value.
#
# #
# Summary:
# Overall, I will recommend the Adaboost model to a potential client to use.
# It is fast and easy to train. It also has the pretty good performance among
# these 3 models. Moreover, it performs OK even we don't have lots of data.
# Even it takes lots of time to train, it is still easy to train,
# compared with Forest which I need to try different parameter combinations.
#
#
# #
# The folowing table is running on 39992 training data, and 960 testing data
# after all parameter tuning.
# Model          Accuracy        Training Time        Classify Time
# Nearest K      0.7167          1s                   >1000s
# Adaboost       0.6938          400s                 0.1s
# Forest         0.6604          30s                  2s
# Best(Adaboost) 0.6938          400s                 0.1s
#
# Sampling: test image easily misclassified.
# 10196604813.jpg
# 12178947064.jpg
# 14965011945.jpg
# 16505359657.jpg
# 17010973417.jpg
# 19391450526.jpg
# 3926127759.jpg
# 4813244147.jpg
# 9151723839.jpg
# 9356896753.jpg
#
# For those misclassified, some of them are too dark (19391450526.jpg);
# some of them are lack of color contrast (2091717624.jpg);
# some of them are hard to distinguish even for human (16505359657.jpg).
# However, most misclassified images share no common pattern at all,
# so I think using color as features may not easily for human to reason about.
# #
#### End Report

#### Model Parameter

### For nearest k
NEAREST_K = 10

### For adaboost
NUM_ADABOOST_CLASSIFIER = 16

### For forest
# the number of tree in forest
NUM_OF_TREE = 150
DATA_PER_TREE = 300

# Avoid overfitting
MAX_TREE_DEPTH = 22

# How many time we try to sample idx to split the data
SPLIT_SAMPLING = 60


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
    pred_ret_1 = {'0': 0, '90': 0, '180': 0, '270': 0}
    pred_ret_2 = {'0': 0, '90': 0, '180': 0, '270': 0}
    pred_ret_3 = {'0': 0, '90': 0, '180': 0, '270': 0}
    for d in dataset:
        if d["data"][idx] < 85:
            pred_ret_1[d["label"]] += 1
        elif d["data"][idx] > 170:
            pred_ret_3[d["label"]] += 1
        else:
            pred_ret_2[d["label"]] += 1

    best_label_1 = max(pred_ret_1, key=pred_ret_1.get)
    best_label_2 = max(pred_ret_2, key=pred_ret_2.get)
    best_label_3 = max(pred_ret_3, key=pred_ret_3.get)
    data_in_1 = sum(pred_ret_1.values())
    data_in_2 = sum(pred_ret_2.values())
    data_in_3 = sum(pred_ret_2.values())
    # print("====")
    # print(pred_ret_1, best_label_1)
    # print(pred_ret_2, best_lt_label)

    # This idx is not useful if we get both same label on two side
    if best_label_1 == best_label_2 or best_label_2 == best_label_3:
        return (1, idx, best_label_1, best_label_2, best_label_3)

    error_num_in_1 = 0
    error_num_in_2 = 0
    error_num_in_3 = 0

    if data_in_1 > 0:
        error_num_in_1 = (
            data_in_1 - pred_ret_1[best_label_1]) * 1.0 / data_in_1
    if data_in_2 > 0:
        error_num_in_2 = (
            data_in_2 - pred_ret_2[best_label_2]) * 1.0 / data_in_2
    if data_in_3 > 0:
        error_num_in_3 = (
            data_in_3 - pred_ret_3[best_label_3]) * 1.0 / data_in_3

    entropy = 0
    if error_num_in_1 > 0:
        entropy += (data_in_1 * 1.0 /
                    total) * -error_num_in_1 * math.log(error_num_in_1)
    if error_num_in_2 > 0:
        entropy += (data_in_1 * 1.0 /
                    total) * -error_num_in_2 * math.log(error_num_in_2)
    if error_num_in_3 > 0:
        entropy += (data_in_1 * 1.0 /
                    total) * -error_num_in_3 * math.log(error_num_in_3)

    return (entropy, idx, best_label_1, best_label_2, best_label_3)


def split_data_by_idx(dataset, idx):
    data_in_1 = []
    data_in_2 = []
    data_in_3 = []

    for d in dataset:
        if d["data"][idx] < 85:
            data_in_1.append(d)
        elif d["data"][idx] > 170:
            data_in_3.append(d)
        else:
            data_in_2.append(d)

    return (data_in_1, data_in_2, data_in_3)


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

        (entropy, best_idx, best_label_1, best_label_2,
         best_label_3) = best_result

        # append selected idx to set
        new_choosed_idx = choosed_idx[:]
        choosed_idx.append(best_idx)

        (data_in_1, data_in_2, data_in_3) = split_data_by_idx(
            dataset,
            best_idx,
        )
        return {
            "idx": best_idx,
            "data_1": build_tree(data_in_1, choosed_idx),
            "data_2": build_tree(data_in_2, choosed_idx),
            "data_3": build_tree(data_in_3, choosed_idx),
        }
    else:
        # we return the prediction based on the most freq label
        return best_label


def tree_classify(tree, data):
    if isinstance(tree, basestring):
        return tree
    if data["data"][tree["idx"]] < 85:
        return tree_classify(tree["data_1"], data)
    elif data["data"][tree["idx"]] > 170:
        return tree_classify(tree["data_3"], data)
    else:
        return tree_classify(tree["data_2"], data)


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