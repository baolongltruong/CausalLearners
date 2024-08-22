import pandas as pd
import numpy as np
import metrics
from econml.metalearners import XLearner
from econml.dr import DRLearner
from econml.dml import CausalForestDML
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

def simulation_simple(n, p, beta, sigma):
    """
    Simulate data for the simple model using pandas.
    
    Parameters:
    n (int): Number of individuals
    p (int): Number of covariates
    beta (float): Coefficient for the treatment effect
    sigma (float): Standard deviation of the noise term
    
    Returns:
    pd.DataFrame: DataFrame containing covariates, treatment indicator, treatment effect, and outcome
    """
    
    # Covariates X_{ij} ~ N(0, 1)
    X = np.random.normal(0, 1, (n, p))
    
    # Create DataFrame with covariates
    df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(p)])
    
    # Treatment effect for individual i: \tau_i = \beta * x_{i1} 
    df['tau'] = beta * df['X1']
    
    # Random treatment indicator: Z_i ~ Bernoulli(0.5)
    df['Z'] = np.random.binomial(1, 0.5, n)
    
    # Outcome: Y_i = Z_i * \tau_i + \epsilon, where \epsilon ~ N(0, \sigma^2)
    df['epsilon'] = np.random.normal(0, sigma**2, n)
    df['Y'] = df['Z'] * df['tau'] + df['epsilon']
    
    return df

def simulation_categorical(n, p, beta, sigma):
    """
    Simulate data for the categorical model using pandas.
    
    Parameters:
    n (int): Number of individuals
    p (int): Number of covariates
    beta (float): Coefficient for the treatment effect
    sigma (float): Standard deviation of the noise term
    
    Returns:
    pd.DataFrame: DataFrame containing covariates, treatment indicator, treatment effect, and outcome
    """
    
    # Covariates X_{ij} ~ N(0, 1)
    X = np.random.normal(0, 1, (n, p))
    
    # Create DataFrame with covariates
    df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(p)])
    
    # Treatment effect for individual i:  \tau_i = \beta * \1\{x_{i1} > 0\}
    df['tau'] = beta * np.where(df['X1'] > 0, 1, 0)
    
    # Random treatment indicator: Z_i ~ Bernoulli(0.5)
    df['Z'] = np.random.binomial(1, 0.5, n)
    
    # Outcome: Y_i = Z_i * \tau_i + \epsilon, where \epsilon ~ N(0, \sigma^2)
    df['epsilon'] = np.random.normal(0, sigma**2, n)
    df['Y'] = df['Z'] * df['tau'] + df['epsilon']
    
    return df

def Causal_LR(data):
    lr_xfit = data.copy()
    lr_xfit['X1*Z'] = lr_xfit['X1'] * lr_xfit['Z'] #Setting interaction term
    lr_xfit = lr_xfit[['X1', 'Z', 'X1*Z']]
    
    lr = LinearRegression() #Fit linear regression
    lr.fit(lr_xfit, data['Y'])
    
    bz = lr.coef_[1]
    bzx = lr.coef_[2]
    
    tau_hat_lr = bz + bzx*data['X1']
    return tau_hat_lr

def Causal_XLearner(data, models):
    X = data[[col for col in data.columns if col.startswith('X')]]
    T = data['Z'] #treatment indicator
    y = data['Y']
    est = XLearner(models=models)
    est.fit(y, T, X=X)
    tau_hat_x = est.effect(X)
    return tau_hat_x

def Causal_DRLearner(data):
    X = data[[col for col in data.columns if col.startswith('X')]]
    T = data['Z'] #treatment indicator
    y = data['Y']
    est = DRLearner()
    est.fit(y, T, X=X, W=None)
    tau_hat_dr = est.effect(X)
    return tau_hat_dr

def Causal_CausalForest(data):
    X = data[[col for col in data.columns if col.startswith('X')]]
    T = data['Z'] #treatment indicator
    y = data['Y']
    est = CausalForestDML(discrete_treatment=True)
    est.fit(y, T, X=X, W=None)
    tau_hat_cf = est.effect(X)

    return tau_hat_cf
