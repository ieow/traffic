import torch
import pandas as pd
import numpy as np
import os

import geohash

DIR = './'

def decodegeo(geo, which):
  if len(geo) >= 6:
    geodecoded = geohash.decode(geo)
    return geodecoded[which]
  else:
    return 0

def convert_timestamp (data) :
  # print(data)
  x = data.split(':')
  data = int(x[0]) * 4 + int(x[1])/15
  return int(data)

def preprocess (data_path) :
  tf_data = pd.read_csv(data_path, index_col=None, header=0)

  # fix timestamp column
  tf_data['timestamp'] = tf_data['timestamp'].apply (convert_timestamp) 
  print(tf_data.iloc[0]['geohash6'] )

  # get geohash precision
  bbox = geohash.bbox(tf_data.iloc[0]['geohash6'])
  long_precision = (bbox['n'] - bbox['s']) / 2
  lat_precision = (bbox['e'] - bbox['w']) / 2

  # unhash geohash6
  # Get the longitude and latitude from the geohash
  tf_data['latitude'] = tf_data['geohash6'].apply(lambda geo: decodegeo(geo, 0))
  tf_data['longitude'] = tf_data['geohash6'].apply(lambda geo: decodegeo(geo, 1))

  long_max = tf_data['longitude'].max()
  long_min = tf_data['longitude'].min()
  long_range = long_max - long_min

  lat_max = tf_data['latitude'].max()
  lat_min = tf_data['latitude'].min()
  lat_range = lat_max - lat_min

  mat_y_max = (long_range / long_precision).astype(int) + 1
  mat_x_max = (lat_range / lat_precision).astype(int) + 1

  print(long_precision)
  print(lat_precision)

  print(long_max)
  print(lat_max)
  
  print(mat_x_max)
  print(mat_y_max)
  
  tf_data['latitude'] = (( tf_data['latitude'] - lat_min ) / lat_precision).astype('int')
  tf_data['longitude'] = (( tf_data['longitude'] - long_min ) / long_precision).astype('int')

  def matrixfy (data) :
    matrix = np.zeros((mat_x_max, mat_y_max))

    for index, row in data.iterrows():
      matrix[row['latitude'], row['longitude']] = row['demand']
    return matrix

  #sort and group
  tf_data_mat = tf_data.sort_values(['day','timestamp']).groupby(['day','timestamp']).apply(matrixfy).unstack()
  tf_data_mat = tf_data_mat.reset_index()

  # tf_data_mat.to_csv('./processed_data.csv')
  # np.save('./process_data_pt',tf_data_mat.values )

  print(tf_data_mat.columns)
  print(tf_data_mat['day'])

  tf_data_melt = tf_data_mat.melt( id_vars=['day'], var_name='timestamp', value_name='demand_map').sort_values(['day','timestamp'])

  # tf_data_melt.to_csv('./data_melt.csv')
  # np.save('./data_melt_pt',tf_data_melt['demand_map'].values )
  print(tf_data_melt)

  # fill nan with witn npzero
  npzero = np.zeros((mat_x_max, mat_y_max))
  def replace_nan (data) : 
    if np.isnan(data).any() :
      print(data)
      data = npzero
    return data

  tf_data_melt['demand_map'] = tf_data_melt['demand_map'].apply(replace_nan)


  # process time
  tf_data_melt['day'] = tf_data_melt['day'] % 7
  tf_data_melt['day'] = tf_data_melt['day'] / 7
  
  tf_data_melt['timestamp'] = tf_data_melt['timestamp'] / 96

  return tf_data_melt


if __name__ == "__main__": 
  # Get config for sequence generation
  data_path = '~/Desktop/grab/traffic-management/Traffic Management/training.csv'
  data = preprocess(data_path)
  # data.to_csv('./data_melt.csv')

  # reorganize to np data
  print('saving data')
  npdata = data['demand_map'].values
  npdata = np.array(list(npdata), dtype= np.float)
  np.save( DIR + '/demand_map', npdata )
  print ( f'demand map shape : {npdata.shape}')

  tp_data = data[ ['day', 'timestamp']].values
  tp_data = np.array(tp_data, dtype= np.float)
  np.save( DIR + '/tp_data' , tp_data)
  print ( f'tp data shape : {tp_data.shape}')
  print('done')

