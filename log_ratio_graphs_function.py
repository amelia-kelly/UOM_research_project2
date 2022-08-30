### Log-likelihood graphs ###


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.stats import lognorm


#Takes the mean, sd, the number of bins, the desired parameter, the conversion back to the original unit (from mmol given by (COMETS), the min value for x, scaling the x, scale for the width (so they dont overlap) & labels for the plot.
#Gives graph with the pdf and enrichment and depletions (i.e. log(o/e)).

def log_ratio(mu, sigma, bin_no, param_name, unit_conversion, min_val, x_scale, w_scale, x_label, title, save_name):
    
    ### observed values ###
    obs = closest[param_name] #in mmol
    obs = obs*unit_conversion #in desired unit (back to the original unit the priors were in)
    
    
    ### Distribution ###
    x1 = np.logspace(min_val, np.log10(max(obs)), 1000)
    pdf = (np.exp(-(np.log(x1) - mu)**2 / (2 * sigma**2))
         / (x1 * sigma * np.sqrt(2 * np.pi)))
    
    
    #log spaced bins to assign observations to
    bins = np.logspace(min_val, np.log10(max(obs)), bin_no)
    
    
    #data
    x2 = bins


    w = bins/w_scale #make smaller to prevent overlap
    obs_count =[]
    exp_count = []
    s=min_val

    #assigns obs to bins and finds cdf of bins
    for b in bins:
        obs_count.append(len(obs[obs<=b])-len(obs[obs<s]))
        cdf1 = 0.5*(1+math.erf((np.log(b)-mu)/(sigma*np.sqrt(2))))
        cdf2 = 0.5*(1+math.erf((np.log(s)-mu)/(sigma*np.sqrt(2))))
        exp_count.append((cdf1-cdf2)*len(obs))
        s=b


    #calculates log ration

    h =[] #height of bars
    for i in range(0,len(bins)):
        o = obs_count[i] + 1
        e = exp_count[i] + 1
        h.append(np.log10(o/e)) #bar height is the log-ratio
   
    xnew = x2/x_scale
    wnew = w/x_scale
    
    #plots the pdf - y-axis on the left
    fig, ax1 = plt.subplots()
    ax1.plot(x1, pdf, linewidth=2, color='r')
    ax1.set_ylabel("Likelihood", fontsize = 17)

    plt.xscale('log')
    plt.xlabel(x_label, fontsize = 17)
    plt.tick_params(labelsize=13)

    
    #plots log-ratios - y-axis on the right
    ax2 = ax1.twinx()
    ax2.set_ylabel("Enrichment/Depletion", fontsize = 17)

    ax2.bar(xnew, height = h, width = wnew, align='edge', alpha = 0.8, color = 'purple')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth = 0.8)

    # Adding title & formatting graphs
    plt.title(title, fontweight ="bold", fontsize = 17)
    plt.tight_layout()
    
    x_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax1.xaxis.set_minor_locator(x_minor)
    ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    
    plt.savefig(save_name)
    plt.show()









