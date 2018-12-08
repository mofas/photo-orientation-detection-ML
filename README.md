# Design and implemntation

### Nearest model:

Nearest model is very simple, we just look through all the images,
and I use "vector_diff" function to find the accumulated square vector difference between
two images in all 192 pixels. The only parameter we need to decide is how many "closest"
images (NEAREST_K) to choose to vote for the final result.
I try K = 5, K = 10, k = 20, and k = 50

```
K = 5 : 0.5979
K = 10 : 0.6177
K = 50 : 0.6260
K = 100: 0.6333
```

However, the performance of the nearest model decrease significantly if we don't have enough
data. For example, if we only have 1000 training image, the accuracy is lower than 0.3,
which is pretty close random guess.

Another drawback of the nearest model is its classified time take too long,
because we actually compared the test image to all the training data.

In short, nearest k model performed well when you have lots of data. However,
the extremely long classify time to make this algorithm impractical.

### Adaboost model:

In AdaBoost model, I iterate all the possible weak classifier candidates (192\*192)
and choose the best n classifier (NUM_ADABOOST_CLASSIFIER) to form the
final AdaBoost classifier. In addition, I find the lots of candidates are very
good at label "0" images. While only a few candidates can do good jobs at label "270".

Therefore, I choose the best NUM_ADABOOST_CLASSIFIER/4 candidates for all orientations.
I believe through this mechanism, I can find the "experts" for all orientations.

The following is the accuracy for different NUM_ADABOOST_CLASSIFIER.

```
NUM_ADABOOST_CLASSIFIER = 2 : 0.2646
NUM_ADABOOST_CLASSIFIER = 5 : 0.659375
NUM_ADABOOST_CLASSIFIER = 10 : 0.678125
NUM_ADABOOST_CLASSIFIER = 16 : 0.6927
NUM_ADABOOST_CLASSIFIER = 20 : 0.6927
NUM_ADABOOST_CLASSIFIER = 28 : 0.6271
NUM_ADABOOST_CLASSIFIER = 50 : 0.553125
NUM_ADABOOST_CLASSIFIER = 500 : 0.575
```

You can see here that increase the number of weak classifiers will not increase
the performance, even decrease it.
I think I can explain that if there are too many too weak classifiers can vote,
then it will create lots of noise and decrease the performance.

The classify time of AdaBoost is blazing fast because we just summary the
result of all classifiers.

Another benefit of AdaBoost is it works quite well even we don't have lots of data.
For 1000 data, the performance of AdaBoost can over 0.4, which is pretty good
compared with the nearest K.

However, the training time of AdaBoost is a drawback, for training around 40k
data, it will take around 400 sec in my computer to train, which is much slower
than forest model.

### Forest model:

Forest model is quite complex to implement because lots of decisions to make
and parameters to tune.
In my implementation, I use the value of a pixel as features to split dataset.
If the value of certain pixel higher than parameter THRESHOLD, then
we put them into the left subtree, otherwise, in the right subtree.

In addition, when we split the data, which index to use is also important.
In theory, we can try all 192 indexes and choose the one giving us the lowest entropy.
However, I find if I randomly choose some good indexes but not the best.
I will get the better performance in the end!

This is an amazing result. Therefore, I have a parameter called "SPLIT_SAMPLING"
to decide how many samplings we did for finding a good index.

Another experiment I try it how many data per tree. Notice, if the number of tree
multiple by the number of data per tree larger than the total data we have,
we will need to "reuse" some data. I do the several experiments and I find if
a data is reused in more than 2 trees, the performance will decrease
gradually.

Other parameter is how many trees we should have and how depth each tree will be.
It is very hard to tune. Nevertheless, both of them should depend on how many
train data you have. From my personal experience, if we give a tree too many data,
then it will grow very high(deep), which caused the overfitting. If we give too few
data, then the tree is underfitting. Similarity, growing too many trees or too few
trees will decrease performance.

I try the several parameters, and I find the optimal max tree depth is around 44
(depend on how many data you feed into the tree),
and the optimal tree number is around 500 ~ 800. The optimal threshold for
splitting data is around 130 ~ 170. The number of data feeds to each tree is
around 60. The number of sampling split index is around 50 ~ 100.

The forest model has several advantages. First of all, relatively faster training
time compared to AdaBoost, which help developers to tune parameters easily.
Second, the classified time is also very fast, in just several seconds.
Third, forest model doing quite well even we don't have lots of data.
For 1000 data, the accuracy of forest model can over 0.55 if we tune the
parameter properly. Compared with 0.3 of nearest K, and 0.4 of AdaBoost,
the performance is unbelievably good.

However, the main disadventage of the forest model is that we need to
tune the parameters by hand, which is a time consuming job. The worst of all,
the performance landscape is quite rough, there is no easy way to run
gradient descent algorithm to find the optimal value.

### Summary:

Overall, I will recommend the Adaboost model to a potential client to use.
It is fast and easy to train. It also has the pretty good performance among
these 3 models. Moreover, it performs OK even we don't have lots of data.
Even it takes lots of time to train, it is still easy to train,
compared with Forest which I need to try different parameter combinations.

The folowing table is running on 39992 training data, and 960 testing data
after all parameter tuning.
Model Accuracy Training Time Classify Time
Nearest K 0.7167 1s >1000s
Adaboost 0.6938 400s 0.1s
Forest 0.6604 30s 2s
Best(Adaboost) 0.6938 400s 0.1s

# Sampling: test image easily misclassified.

```
10196604813.jpg
12178947064.jpg
14965011945.jpg
16505359657.jpg
17010973417.jpg
19391450526.jpg
3926127759.jpg
4813244147.jpg
9151723839.jpg
9356896753.jpg
```

For those misclassified, some of them are too dark (19391450526.jpg);
some of them are lack of color contrast (2091717624.jpg);
some of them are hard to distinguish even for human (16505359657.jpg).
However, most misclassified images share no common pattern at all,
so I think using color as features may not easily for human to reason about.
