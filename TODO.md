# Test command

train

```
time ./orient.py train train-data.txt adaboost_model.txt adaboost
time ./orient.py train train-data.txt forest_model.txt forest
```

test

```
time ./orient.py test test-data.txt adaboost_model.txt adaboost
time ./orient.py test test-data.txt forest_model.txt forest
```

# Write report

### Which classifiers and which parameters would you recommend to a potential client?

### How does performance vary depending on the training dataset size,

### Show a few sample images that were classified correctly and incorrectly.

# read args from argv

Test data
nearest: 0.6260416666666667
adaboost: 0.6927
forest: 0.7302

adaboost:
forest:
best THRESHOLD: 180

# train model based on the real input

best_model.txt.
