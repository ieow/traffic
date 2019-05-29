import numpy 
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 

import glob

path = '/home/apollo/Desktop/grab/safety/safety/features'
all_files = glob.glob(path + "/*-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
print(all_files)
li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

train = pd.concat(li, axis=0, ignore_index=True)

labels = pd.read_csv('~/Desktop/grab/safety/safety/labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv')
# labels  = labels.sort_values(by=['bookingID'])

# Remove error data




data_agg = train.groupby('bookingID', as_index=False).agg({
  "acceleration_x" : [min,max,"mean"],
  "acceleration_y" : [min,max,"mean"],
  "acceleration_z" : [min,max,"mean"],
  "gyro_x" :  [min,max,"mean"],
  "gyro_y" :  [min,max,"mean"],
  "gyro_z" :  [min,max,"mean"],
  "Speed"  :  [min,max,"mean"]
})
# data_agg.columns = data_agg.columns.droplevel(level=0)
data_agg.columns = data_agg.columns.map('_'.join)

data_agg = data_agg.rename(columns = { "bookingID_" : "bookingID"})
# print(data_agg.columns)

df_merged = pd.merge(  data_agg, labels,  on='bookingID', how='inner')
# df_merged = df_merged.dropna()

# print(data_agg)
# print(df_merged)


# Drop data with bad gps accuracy ( *max meter 6e3, suggested  )





# Seperate train set and validation set
col = df_merged.columns.difference(['bookingID', 'label']).values
xTrain, xTest, yTrain, yTest = train_test_split( df_merged[col], df_merged['label'], test_size = 0.3, random_state = 0)

# Massage
# x_mean = xTrain.mean(axis='index')
# x_std  = xTrain.std()

# xTrain = xTrain - x_mean
# xTrain = xTrain / x_std
print(xTrain.columns)


# xTest = xTest - x_mean
# xTest = xTest / x_std

algo = ''
# Train
# Random Forest
if (algo =='RandomForest') :
  clf = RandomForestClassifier(n_jobs=2, random_state=0)
  clf.fit(xTrain , yTrain )

  pred = clf.predict(xTest)

#CatBoost
else :
  model = CatBoostClassifier( iterations=200, random_seed=0, verbose=True , task_type = "GPU")
  model.fit(xTrain, yTrain )
  print(model.feature_importances_)
  print(model.best_score_)

  pred = model.predict(xTest)



# pd.concat([test['PassengerId'], pd.DataFrame(pred, columns=['Survived'])], axis=1).to_csv('submission.csv', index=False)
tn, fp, fn, tp = metrics.confusion_matrix(yTest, pred ).ravel()
f1_score = metrics.f1_score( yTest, pred ) 
accuracy_score = metrics.accuracy_score( yTest, pred ) 

print( f"tn={tn}, fp={fp}, fn={fn}, tp={tp}" )
print ( f1_score)
print( accuracy_score )


# dataset = numpy.array([[1,4,5,6],[4,5,6,7],[30,40,50,60],[20,15,85,60]])
# train_labels = [1.2,3.4,9.5,24.5]
# model = CatBoostRegressor(learning_rate=1, depth=6, loss_function='RMSE')
# fit_model = model.fit(dataset, train_labels)

# print(fit_model.get_params())

# ['Speed_max', 'Speed_mean', 'Speed_min', 'acceleration_x_max',
#        'acceleration_x_mean', 'acceleration_x_min', 'acceleration_y_max',
#        'acceleration_y_mean', 'acceleration_y_min', 'acceleration_z_max',
#        'acceleration_z_mean', 'acceleration_z_min', 'gyro_x_max',
#        'gyro_x_mean', 'gyro_x_min', 'gyro_y_max', 'gyro_y_mean', 'gyro_y_min',
#        'gyro_z_max', 'gyro_z_mean', 'gyro_z_min'],