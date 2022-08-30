### Sim points function ###

import pandas as pd
import numpy as np

#returns a list of data points for each respective simulation
#can then be used with sci-kit learns (v.1.8.1) K-medoid function
def sim_points(df, result):
    #create a list of arrays that contain each simulations time points (801 dimensions)
    sim_list = []
    sim_names = df['Simulation'].unique()

    for i in range(0, len(sim_names)):
        sim_name = sim_names[i]
        sim_find = df.loc[(df['Simulation'] == sim_name)][[result]]
        sim_array = sim_find[result].values
        sim_list.append(sim_array)

    #convert from a list to an array of each point
    sim_list = np.stack(sim_list)
    return sim_list

