# model is nearest, adaboost, forest, or best
import sys
import heapq
import numpy as np


def vector_diff(img_data1, img_data2):
    diff = 0
    for i in range(192):
        diff += (img_data1["data"][i] - img_data2["data"][i])**2
    return diff


def general_extract_image_data(img_info):
    data = img_info.split(' ')
    return {
        'filename': data[0],
        'label': data[1],
        'data': [int(x) for x in data[1:]]
    }


# for nearest model
def nearest_extract_image_data(img_info):
    data = img_info.split(' ')
    return {'label': data[0], 'data': [int(x) for x in data[1:]]}


# img1 = "train/6179019779.jpg 180 142 135 134 148 141 139 146 139 137 134 127 122 126 121 119 87 81 87 107 124 155 112 132 167 135 128 125 115 108 102 104 97 94 111 108 110 110 124 149 123 141 172 122 140 173 120 139 173 126 120 116 108 113 126 103 116 141 127 143 173 128 148 181 129 148 181 129 148 183 127 146 184 143 155 178 142 160 191 139 157 188 130 147 177 110 131 163 126 147 180 143 162 196 143 162 197 91 114 149 93 115 150 86 108 141 70 92 125 61 85 119 66 93 128 106 132 167 137 161 197 83 110 148 92 117 154 98 123 158 92 118 154 94 117 148 97 124 155 88 119 160 91 122 164 201 212 227 196 209 225 199 213 228 204 216 229 207 219 232 203 218 233 176 194 215 173 190 212 197 210 231 177 197 225 172 194 225 189 206 231 200 214 235 198 214 237 213 226 244 218 229 245"
# img2 = "train/6143621243.jpg 0 151 178 209 131 169 210 147 176 211 172 184 211 125 162 208 100 149 209 99 145 207 98 142 204 180 203 227 184 208 231 179 206 230 166 200 232 150 195 235 115 173 225 99 162 222 89 152 219 223 232 234 220 231 236 217 237 247 190 220 236 117 146 170 159 198 228 143 188 226 123 175 220 238 244 240 246 255 255 233 235 236 136 130 135 119 116 136 223 241 253 213 230 239 205 221 232 202 181 149 216 193 171 218 136 122 207 109 93 206 148 123 222 215 174 211 199 168 229 208 168 88 79 76 130 84 88 167 106 108 111 80 75 84 77 59 96 77 86 86 70 78 63 57 42 33 39 17 40 42 18 42 45 22 45 48 22 40 42 24 46 44 41 33 38 22 15 27 7 19 29 8 24 33 8 28 35 8 26 35 9 28 35 8 23 32 4 28 36 13 27 34 18"

# img_data1 = extract_image_data(img1)
# img_data2 = extract_image_data(img2)

model_nearest = []
model_adaboost = []
model_forest = []
model_best = []

# Hyperparameter
NEAREST_K = 10


def nearest_classify(test_data):
    # find nearest K
    # we need priority queue to keep the best K cand in the queue

    cands = []
    pred_ret = {}
    for cand in model_nearest:
        diff = vector_diff(cand, test_data)
        heapq.heappush(cands, (-diff, cand["label"]))
        if len(cands) > NEAREST_K:
            removed = heapq.heappop(cands)
            # print("removed", removed)

    # get the result
    while len(cands):
        (score, pred) = heapq.heappop(cands)
        if pred not in pred_ret:
            pred_ret[pred] = 1
        else:
            pred_ret[pred] += 1
        # print("in", (score, pred))

    # print(pred_ret, max(pred_ret, key=pred_ret.get))
    return max(pred_ret, key=pred_ret.get)


def train(input_file, output_file, model):
    with open(input_file, 'r') as file:
        train_data = file.readlines()

    # in nearest, we just rmeove the filename from data
    if model == 'nearest':
        with open(output_file, 'w') as file:
            for row in train_data:
                file.write(" ".join(row.split(' ')[1:]))

    return


def test(test_file, model_file, model):

    if model == 'nearest':
        with open(model_file, 'r') as file:
            model_data = file.readlines()
        for row in model_data:
            model_nearest.append(nearest_extract_image_data(row))

        with open(test_file, 'r') as file:
            test_data = file.readlines()

        results = []
        for row in test_data:
            results.append(nearest_classify(general_extract_image_data(row)))
            break

    return


# train("/Users/cyli/code/cli3-a4/train-data-s.txt",
#       "/Users/cyli/code/cli3-a4/nearest_model.txt", "nearest")

test("/Users/cyli/code/cli3-a4/test-data-s.txt",
     "/Users/cyli/code/cli3-a4/nearest_model.txt", "nearest")

# task_type = sys.argv[1]

# if task_type == "train":
#     train(sys.argv[2], sys.argv[3], sys.argv[4])
# elif task_type == "test":
#     test(sys.argv[2], sys.argv[3], sys.argv[4])