# traffic
Traffic Management Challenge 


**To Train**
- Create a log/saving dir 
```
mkdir \<dir\>
```
- Preprocess data 
```
python traffic/dataprocess.py --logdir <dir>
```

- Train model 
```
python traffic/trainpred.py --logdir <dir>
```

**To Test**
```
python traffic/trainpred.py --logdir <dir> --test --data_path <data_path>
```
**To Run Pred**
```
python traffic/trainpred.py --logdir <dir> --prediction --data_path <data_path>
```
