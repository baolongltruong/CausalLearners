import sys 
import os
import argparse
import json
import time
import gc

sys.path.append(os.path.join(os.getcwd(),'Modules'))
curdir = os.chdir('Scripts')

from causalsim import *
import metrics


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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

    #Timing each learner
    execution_times = {learner: [] for learner in learners}
    
    for i in range(num_sim):
        data = eval(data_str)
        tau = np.array(data['tau'])

        for learner in learners:
            start_time = time.perf_counter()
            tau_hat = eval(learners[learner])
            metric_i = metrics.evaluate(tau, tau_hat)

            #Bug with DR-Learner where predictions are way off, happens randomly even for same dataset. 
            if metric_i[0] < -50000 or metric_i[0] > 50000:
                continue
                
            metrics_result[learner]['mse'].append(metric_i[0])
            metrics_result[learner]['bias'].append(metric_i[1])
            metrics_result[learner]['r2'].append(metric_i[2])

            #Timing
            end_time = time.perf_counter()
            execution_times[learner].append(end_time - start_time)
        
        #Garbage collection
        gc.collect()
        
    
    return metrics_result, execution_times


def plot_metric(metric_name, res, title_label, xlabel, file_prefix, log = True):
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
        errors = [] 
        for n in iv:
            metric_data = res[n][model][metric_name]
            metric_values.append(np.mean(metric_data))  # Taking the mean of the list
            errors.append(np.std(metric_data) / np.sqrt(len(metric_data)))
       
        # Plot each model's metrics against the iv
        plt.errorbar(iv, metric_values, yerr=errors, label=model, marker='o', capsize=5)
        #plt.plot(iv, metric_values, label=model, marker='o')
    
    # Labeling the plot
    plt.title(f'{metric_name.upper()} vs {title_label}')
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{metric_name.upper()}')
    plt.legend(title='Model')
    plt.grid(True)
    if log:
        plt.xscale('log')  # Optional: Log scale if IV varies greatly
    #plt.show()
    
    if not os.path.exists(f'results/{file_prefix}'):
        os.makedirs(f'results/{file_prefix}')
    plt.savefig(f'results/{file_prefix}/{file_prefix}_{metric_name}.png', dpi=300)


if __name__ == '__main__':
    #Parse config path argument
    
    parser = argparse.ArgumentParser(description="Process a configuration file.")
    parser.add_argument(
        'config_path',
        type=str,
        help="Path to the configuration JSON file."
    )
    args = parser.parse_args()
    config_path = args.config_path
    
    with open(config_path, 'r') as file:
        config = json.load(file)
    print('Config loaded...')

    #Run experiment
    start_time = time.time()
    
    res = {}
    time_res = {}
    for iv in config['iv_list']:
        print(f"Run: {iv}")
        res[iv], time_res[iv] = run_experiment(config['learners'], config['data_str'], config['num_sim'])
    
    for metric in ['mse', 'bias', 'r2']:
        plot_metric(metric, res, config['iv_name'], config['iv_label'], config['test_name'], config['log'])
        
    end_time = time.time()
    
    #Print metric results
    '''
    models = list(next(iter(res.values())).keys())
    iv = list(res.keys())
    with open(f"results/{config['test_name']}/results.txt", 'w') as f:  # Open the file in write mode
        for n in iv:
            for metric in ['mse', 'bias', 'r2']:
                for model in models:
                    mean_value = np.mean(res[n][model][metric])
                    std_value = np.std(res[n][model][metric])
                    print(f"{config['iv_name']}: {n}, Model: {model}, Metric: {metric} | Mean: {mean_value}, STD: {std_value}", file=f)'''
    models = list(next(iter(res.values())).keys())
    iv = list(res.keys())
    
    # Initialize an empty list to collect data
    data = []
    
    # Loop through the results to collect data for the DataFrame
    for n in iv:
        for metric in ['mse', 'bias', 'r2']:
            for model in models:
                mean_value = np.mean(res[n][model][metric])
                std_value = np.std(res[n][model][metric])
                data.append({
                    config['iv_name']: n,
                    'Model': model,
                    'Metric': metric,
                    'Mean': mean_value,
                    'STD': std_value
                })
    
    # Convert collected data into a DataFrame
    df = pd.DataFrame(data)
    df.to_csv(f"results/{config['test_name']}/results.csv", index=False)
    #Print time results
    
    with open(f"results/{config['test_name']}/time.txt", 'w') as f:
        f.write(f"Num Sim: {config['num_sim']}\n\n")
        f.write(f"Total Time: {end_time - start_time}\n\n")
        for iv in time_res:
            mean_execution_times = {learner: np.mean(time_res[iv][learner]) for learner in config['learners']}
            f.write(f"Mean execution times for IV = {iv}\n\n")
            for learner, mean_time in mean_execution_times.items():
                f.write(f"{learner}: {mean_time:.4f} seconds\n")
        
            
