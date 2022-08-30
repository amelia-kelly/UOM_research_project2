##Kolmogorov-Smirnov test with Benjamini-Hochberg correction##

import pandas as pd
import os
import numpy as np
import math
from scipy.stats import kstest
from scipy.stats import ks_2samp



########## KS-test (Massey, 1951) ##########
#takes a dataframe with
##priors and sampled values to determine if they are from the same distribution
##returns a data frame of the p-value and K-S stat for each parameter

def KS_test(priors, sampled):
    params = list(sampled.columns.values)
    
    KS_res = []
    for i in range(0, 10):
        prior = priors.iloc[:, i]
        sample = sampled.iloc[:, i]
        KS = ks_2samp(prior, sample)
        KS_res.append(KS)
        
    results = pd.DataFrame(KS_res)
    KS_results = results.T
    KS_results.columns = [params]
    
    return KS_results


########## BH correction (Benjamini and Hochberg, 1995) ##########
#p-values are then adjusted using a Benjamini-Hochberg correction
#returns a ranked data frame of the K-S stat, p-value, adjusted p-value and rank
def BH_adjust(KS_results, total_trials, FDR):
    results = KS_results.T
    results['rank'] = results['pvalue'].rank()

    ranks = results['rank']
    p_adjust = []
    for x in range(0, len(ranks)):
        adjusted = (ranks[x]/total_trials)*FDR
        p_adjust.append(adjusted)

    results['padjusted'] = p_adjust
    sorted_results = results.sort_values('pvalue')

    return results, sorted_results


