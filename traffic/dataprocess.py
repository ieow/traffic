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
    x = data.split(':')
    data = int(x[0]) * 4 + int(x[1])/15
    return int(data)

class Traffic :

    def __init__(self, dir_path='./', load_config = False ) :
        self.DIR = dir_path
        self.saved_data = False

    def setMapRange(self , tf_data ) :
        self.long_max = tf_data['longitude'].max() 
        self.long_min = tf_data['longitude'].min()
        self.long_range = self.long_max - self.long_min

        self.lat_max = tf_data['latitude'].max()
        self.lat_min = tf_data['latitude'].min()
        self.lat_range = self.lat_max - self.lat_min

        lat, lon, lat_delta, lon_delta = geohash.decode(tf_data.iloc[0]['geohash6'], True)

        self.long_delta = lon_delta *2
        self.lat_delta = lat_delta *2

        self.mat_y_max = (self.long_range / self.long_delta).astype(int) + 1
        self.mat_x_max = (self.lat_range / self.lat_delta).astype(int) + 1
        
        self.data_shape = (self.mat_x_max, self.mat_y_max )


    def geohash_decode (self, tf_data) :
        # unhash geohash6
        # Get the longitude and latitude from the geohash
        tf_data['latitude'] = tf_data['geohash6'].apply(lambda geo: decodegeo(geo, 0))
        tf_data['longitude'] = tf_data['geohash6'].apply(lambda geo: decodegeo(geo, 1))
        return tf_data

    def geohash_encode (self, data) :
        # unhash geohash6
        # Get the longitude and latitude from the geohash
        data['geohash'] = data.apply(lambda x: geohash.encode(x[2], x[1] , 6)  , axis= 1)
        
        return data

    def matrixfy (self, data) :
        matrix = np.zeros((self.mat_x_max, self.mat_y_max))

        for index, row in data.iterrows():
            matrix[row['latitude'], row['longitude']] = row['demand']
        
        return matrix
        
    def preprocess_map (self, tf_data) :
        # fix timestamp column
        tf_data['timestamp'] = tf_data['timestamp'].apply (convert_timestamp) 
        
        tf_data['latitude'] = (( tf_data['latitude'] - self.lat_min ) / self.lat_delta).astype('int')
        tf_data['longitude'] = (( tf_data['longitude'] - self.long_min ) / self.long_delta).astype('int')

        #sort and group
        tf_data_mat = tf_data.sort_values(['day','timestamp']).groupby(['day','timestamp']).apply(self.matrixfy).unstack()
        tf_data_mat = tf_data_mat.reset_index()

        tf_data_melt = tf_data_mat.melt( id_vars=['day'], var_name='timestamp', value_name='demand_map').sort_values(['day','timestamp'])

        # fill nan with witn npzero
        npzero = np.zeros((self.mat_x_max, self.mat_y_max))
        def replace_nan (data) : 
            if np.isnan(data).any() :
                data = npzero
            return data

        tf_data_melt['demand_map'] = tf_data_melt['demand_map'].apply(replace_nan)

        return tf_data_melt

    def preprocess_timestamp (self, data) :
        # process time
        data['day'] = data['day'] % 7
        data['day'] = data['day'] / 7
        
        data['timestamp'] = data['timestamp'] / 96

        return data

    def preprocess ( self, data ) :
        print('Preprocessing... Please Wait a while...')
        data = self.preprocess_map(data)
        data = self.preprocess_timestamp(data)
        print('Preprocessing Done')
        return data

    def postprocess ( self , data ) :
        df = pd.DataFrame()
        i=0
        for item in data :
            i+=1
            temp = self.matrixToCoord(item)

            df_temp = pd.DataFrame ( temp, columns=[ 'demand', 'longtitude', 'latitude' ])
            df_temp = self.geohash_encode(df_temp)
            df_temp = df_temp.set_index('geohash')

            if len(df) == 0 :
                # df = df_temp
                df['T+1'] = df_temp['demand']
                # df.drop(['demand', 'longtitude', 'latitude'])
                # df?

            else :
                df[f'T+{i}'] = df_temp['demand']

        return df

    def matrixToCoord (self, data) :
        total_coord = self.mat_x_max * self.mat_y_max 
        print(total_coord)
        # temp = np.zeros( (total_coord ,3 ) )
        temp = []
        for i in range(self.mat_x_max) :
            for j in range(self.mat_y_max) :
                lat = i * self.lat_delta + self.lat_min
                lon = j * self.long_delta + self.long_min
                temp.append( [data[i][j] , lon, lat] )
                # temp[ i*j + j] = [data[i][j] , lon, lat ]
        return temp


if __name__ == "__main__": 
    # Get config for sequence generation

    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--logdir', type=str, help='Directory where results are logged')
    parser.add_argument('--data_path', type=str, default='training.csv', help='Directory where results are logged')

    parser.add_argument('--validate', action='store_true',
                        help='specify test mode')

    args = parser.parse_args()

    data_path = args.data_path

    if not exists(args.logdir) :
        mkdir(args.logdir)

    outfile = join(args.logdir, 'tclass.pkl')

    tclass = Traffic()
    
    read_data = pd.read_csv(data_path, index_col=None, header=0)
    read_data = tclass.geohash_decode(read_data)

    tclass.setMapRange(read_data)

    if not args.validate : 
        tf_data = tclass.preprocess(read_data)
        tclass.saved_data = tf_data

        with open(outfile, 'wb') as pickle_file:
            pickle.dump(tclass, pickle_file )
            pickle_file.close()
        print ( f'Tclass object saved at path {outfile}')

    elif args.validate :

        read_data = read_data[ (read_data['day'] == 1)  ]

        tf_data = tclass.preprocess(read_data)
        tclass.saved_data = tf_data
        
        data_encoded = tclass.matrixToCoord(tf_data.iloc[0]['demand_map'] )
        # print(data_encoded)
        df_data = pd.DataFrame (data_encoded, columns=[ 'demand_map', 'longtitude', 'latitude' ])
        df_data = tclass.geohash_encode(df_data)


        data_time0 = read_data[ (read_data['day'] ==1) & (read_data['timestamp'] == 0) ]
        x = df_data[ df_data['geohash'].isin( data_time0['geohash6'] ) ].sort_values(['geohash']) 

        print(data_time0.sort_values(['geohash6'] ) )
        print(x)

        print( x[x['geohash'] == 'qp09sx'])
        print( data_time0[data_time0['geohash6'] == 'qp09sx'])

        print('testing postprocess')
        data0 = tf_data.iloc[0]['demand_map']
        data1 = tf_data.iloc[1]['demand_map']
        data2 = tf_data.iloc[2]['demand_map']
        data3 = tf_data.iloc[3]['demand_map']

        vdata = np.array( (data0,data1,data2,data3))

        print(vdata.shape)
        
        out = tclass.postprocess(vdata)
        print (out)



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
