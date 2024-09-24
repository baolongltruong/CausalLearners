import sys 
import os

#Directory where script resides in
curdir = os.path.dirname(__file__)

#Adding path to Modules directory
modules_path = os.path.abspath(os.path.join(curdir,'..', 'Modules'))
sys.path.append(modules_path)

from causalsim import *
import metrics

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import uniform

def run_experiment(learners, data_str, num_sim):

    '''n = n  # Number of individuals
    p = p     # Number of covariates
    beta = beta  # Beta_1 value for treatment effect
    sigma = sigma # Sigma value for noise term'''

    metrics_result = {}
    for learner in learners:
            metrics_result[learner] = {'mse': [], 'bias': [], 'r2': []}
        
    
    for i in range(num_sim):
        data = eval(data_str)
        tau = np.array(data['tau'])

        for learner in learners:
            tau_hat = eval(learners[learner])
            metric_i = metrics.evaluate(tau, tau_hat)

            #Bug with DR-Learner where predictions are way off, happens randomly even for same dataset. 
            if metric_i[0] < -500 or metric_i[0] > 500:
                continue
                
            metrics_result[learner]['mse'].append(metric_i[0])
            metrics_result[learner]['bias'].append(metric_i[1])
            metrics_result[learner]['r2'].append(metric_i[2])
            
        
    
    return metrics_result


def plot_metric(metric_name, res, title_label, xlabel, log = True):
    '''
    metric_name = [mse, bias, r2]
    res = result dictionary from run_experiment
    title_label = title of plot
    xlabel = label of x-axis
    log = log(x) for x axis
    '''
    
    models = list(next(iter(res.values())).keys())
    

    iv = list(res.keys())
    
    plt.figure(figsize=(8, 6))  # 

    # Loop through each model and plot the metric for that model across iv
    for model in models:
        metric_values = []
        
        for n in iv:
            metric_values.append(np.mean(res[n][model][metric_name]))  # Taking the mean of the list

       
        # Plot each model's metrics against the iv
     
        plt.plot(iv, metric_values, label=model, marker='o')
    
    # Labeling the plot
    plt.title(f'{metric_name.upper()} vs {title_label}')
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{metric_name.upper()}')
    plt.legend(title='Model')
    plt.grid(True)
    if log:
        plt.xscale('log')  # Optional: Log scale if IV varies greatly
    plt.show()



