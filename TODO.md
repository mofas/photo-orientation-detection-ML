# Test command

train

```
time ./orient.py train train-data.txt nearest_model.txt nearest
time ./orient.py train train-data.txt adaboost_model.txt adaboost
time ./orient.py train train-data.txt forest_model.txt forest
time ./orient.py train train-data.txt best_model.txt best
```

test

```
time ./orient.py test test-data.txt nearest_model.txt nearest
time ./orient.py test test-data.txt adaboost_model.txt adaboost
time ./orient.py test test-data.txt forest_model.txt forest
time ./orient.py test test-data.txt best_model.txt best
```

```
./orient.py train train-data.txt forest_model.txt forest && ./orient.py test test-data.txt forest_model.txt forest
```

# Write report
