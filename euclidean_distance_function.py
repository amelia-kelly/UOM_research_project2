###Euclidean Distances###


import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance


#takes the simulation and experimental data frames 
#1.scales each data point
#2.returns each euclidean distance of each point to experimental data point (exp_df)
def euclidean_dist(df, exp_df, results, weight):
    euc_dist = []
    times = exp_df['time (h)']

    
    for i in range(0, len(exp_df['time (h)'])):
        time = times[i]
        sim_data = df.loc[(df['time (h)'] == time)][['Simulation', 'time (h)', results]] #simulations
        exp_data = exp_df.loc[(exp_df['time (h)'] == time)][['time (h)', results]]
        data = sim_data.append(exp_data, ignore_index = True)

    
        #scaling
        #define min max scaler
        scaler = MinMaxScaler(feature_range=(1, 100))
        #transform data
        scaled = scaler.fit_transform(data[['time (h)', results]])

        #euclidean distances
        for j in range (0, len(scaled)-1): #doesnt include the experimental data which is the last value
            dist = distance.euclidean(scaled[j], scaled[len(sim_data[results])], w = weight)
            euc_dist.append(dist)
    
    return euc_dist


#############################


#takes a dataframe of simulation data, experimental data and data frame of euclidean distances
#then assigns the euclidean distance to the simulation name
def sim_euc_dists(df, exp_df, euc_dist_df):
    #has number of each distance for each simulation used to assign then next
    group = df.groupby(['time (h)'])
    exp_group = exp_df.groupby(['time (h)'])
    times = exp_df['time (h)'].unique()

    idx = []

    for i in range(0, len(times)):
        time = times[i]
        t = group.get_group(time)
        sim = t['Simulation'].nunique()
        idx.append(sim)

    idx.insert(0, 0)
    
    res = []

    #matches up the simulation names to the euclidean distance
    for i in range(0, len(times)):
        time = times[i]
        t = group.get_group(time)
        sim = t['Simulation'].unique()
        dist = euc_dist_df[idx[i]:idx[i]+len(sim)]
        dictionary = {sim[i]: dist[i] for i in range(len(sim))}
        df = pd.DataFrame.from_dict(dictionary, orient='index')
        df.columns=[time]   
    
        res.append(df)

    #joins all sim name-distance pair into one dataframe
    dists = res[0]
    for i in range(0, len(res)-1):
        if i < len(exp_df):
            dists = dists.join(res[i+1])
    
    dists = dists.reset_index()
    dists = dists.rename(columns={'index': 'Simulation'})
        

    return dists


