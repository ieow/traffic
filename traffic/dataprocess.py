import torch
import pandas as pd
import numpy as np
import os
import pickle

import geohash
import argparse
from os.path import join, exists
from os import mkdir

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

class Traffic :

    def __init__(self, dir_path='./', load_config = False ) :
        self.DIR = dir_path
        # self.trainData = 

    def MapToGeohash( x,y ) :
        return 

    def setMapRange(self , tf_data ) :
        # if input is str load config
        
        # else calculate basd of dataframe
        # get geohash precision
        bbox = geohash.bbox(tf_data.iloc[0]['geohash6'])
        self.long_precision = (bbox['n'] - bbox['s']) / 2
        self.lat_precision = (bbox['e'] - bbox['w']) / 2

        self.long_max = tf_data['longitude'].max()
        self.long_min = tf_data['longitude'].min()
        self.long_range = self.long_max - self.long_min

        self.lat_max = tf_data['latitude'].max()
        self.lat_min = tf_data['latitude'].min()
        self.lat_range = self.lat_max - self.lat_min

        self.mat_y_max = (self.long_range / self.long_precision).astype(int) + 1
        self.mat_x_max = (self.lat_range / self.lat_precision).astype(int) + 1
        
        self.data_shape = (self.mat_x_max, self.mat_y_max )

    def geohashToGeocoord (self, tf_data) :
        # unhash geohash6
        # Get the longitude and latitude from the geohash
        tf_data['latitude'] = tf_data['geohash6'].apply(lambda geo: decodegeo(geo, 0))
        tf_data['longitude'] = tf_data['geohash6'].apply(lambda geo: decodegeo(geo, 1))
        return tf_data


    def matrixfy (self, data) :
        matrix = np.zeros((self.mat_x_max, self.mat_y_max))

        for index, row in data.iterrows():
            matrix[row['latitude'], row['longitude']] = row['demand']
            return matrix
        
    def preprocess (self, tf_data) :
        # fix timestamp column
        tf_data['timestamp'] = tf_data['timestamp'].apply (convert_timestamp) 
        # print(tf_data.iloc[0]['geohash6'] )
        
        tf_data['latitude'] = (( tf_data['latitude'] - self.lat_min ) / self.lat_precision).astype('int')
        tf_data['longitude'] = (( tf_data['longitude'] - self.long_min ) / self.long_precision).astype('int')


        #sort and group
        tf_data_mat = tf_data.sort_values(['day','timestamp']).groupby(['day','timestamp']).apply(self.matrixfy).unstack()
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
        npzero = np.zeros((self.mat_x_max, self.mat_y_max))
        def replace_nan (data) : 
            if np.isnan(data).any() :
                # print(data)
                data = npzero
            return data

        tf_data_melt['demand_map'] = tf_data_melt['demand_map'].apply(replace_nan)

        # process time
        tf_data_melt['day'] = tf_data_melt['day'] % 7
        tf_data_melt['day'] = tf_data_melt['day'] / 7
        
        tf_data_melt['timestamp'] = tf_data_melt['timestamp'] / 96

        self.data = tf_data_melt

        return tf_data_melt


    def postprocess ( pred , tf_data ) :
        # for i in data :
        #   for j in i :


        # df = pd.DataFrame(columns= [])
        return


if __name__ == "__main__": 
    # Get config for sequence generation

    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--logdir', type=str, help='Directory where results are logged')
    parser.add_argument('--data_path', type=str, default='training.csv', help='Directory where results are logged')

    args = parser.parse_args()

    data_path = '~/Desktop/grab/traffic-management/Traffic Management/training.csv'

    if not exists(args.logdir) :
        mkdir(args.logdir)

    outfile = join(args.logdir, 'tclass.pkl')

    tclass = Traffic()
    
    tf_data = pd.read_csv(data_path, index_col=None, header=0)
    tf_data = tclass.geohashToGeocoord(tf_data)

    tclass.setMapRange(tf_data)
    tf_data = tclass.preprocess(tf_data)

    print(tf_data)

    with open(outfile, 'wb') as pickle_file:
        pickle.dump(tclass, pickle_file )
        pickle_file.close()


'''
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
'''
