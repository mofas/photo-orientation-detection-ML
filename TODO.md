# Test command

train

```
time ./orient.py train train-data.txt nearest_model.txt nearest
time ./orient.py train train-data.txt adaboost_model.txt adaboost
time ./orient.py train train-data.txt forest_model.txt forest
time ./orient.py train train-data.txt best_model.txt best
```

```
time ./orient.py train train-data-s.txt forest_model.txt forest
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

300 / 150 / 20
0.403125

200 / 200 / 20
0.425

150 / 300/ 25
0.4490

100 / 400 / 30
0.41458333333333336
