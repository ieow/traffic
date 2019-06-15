# traffic
Traffic Management Challenge 


**To Train**
- Create a log/saving dir 
```
mkdir <dir>
```
- Preprocess data 
```
python traffic/dataprocess.py --logdir <dir> --data_path <data_path>
```

- Train model 
```
python traffic/trainpred.py --logdir <dir>
```

- Train model with new data within same geohash range
```
python traffic/trainpred.py --logdir <dir> --data_path <data_path>
```

**To Test**
```
python traffic/trainpred.py --logdir <dir> --test --data_path <data_path>
```

**To Run Pred**
```
python traffic/trainpred.py --logdir <dir> --prediction --data_path <data_path>
```
