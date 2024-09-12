import pandas as pd
import numpy as np
import metrics
from econml.metalearners import XLearner
from econml.metalearners import TLearner
from econml.metalearners import SLearner
from econml.dr import DRLearner
from econml.dml import CausalForestDML
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from scipy.stats import uniform


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

def simulation_XLearner_1(n, p):
    # Covariates X_{ij} ~ N(0, 1)
    X = np.random.normal(0, 1, (n, p))
    
    # Create DataFrame with covariates
    df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(p)])
    
    # τ (x) = 8 I(x2 > 0.1) 
    df['tau'] = 8 * np.where(df['X2'] > 0.1, 1, 0)
    
    # Random treatment indicator: Z_i ~ Bernoulli(0.01)
    df['Z'] = np.random.binomial(1, 0.01, n)
    df['epsilon'] = np.random.normal(0, 1, n)

    #µ0(x) = xTβ + 5 I(x1 > 0.5), with β ∼ Unif [−5, 5]20

    beta = uniform.rvs(-5, 10, size=p) 
    mu0 = np.dot(df[[f'X{i+1}' for i in range(p)]], beta) + 5 * np.where(df['X1'] > 0.5, 1, 0)
    
    df['Y'] = mu0 + df['Z'] * df['tau'] + df['epsilon']
    
    return df

def sig(x):
    return 2 / (1 + np.exp(-12 * (x - 1/2)))

def simulation_XLearner_2(n,p):
    X = np.random.normal(0, 1, (n, p))
    
    # Create DataFrame with covariates
    df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(p)])
    
    # µ1(x) = 1/2ς(x1)ς(x2),µ0(x) = −1/2ς(x1)ς(x2)
    df['tau'] = sig(df['X1'])*sig(df['X2'])

    
    df['Z'] = np.random.binomial(1, 0.5, n)
    df['epsilon'] = np.random.normal(0, 1, n)


    # µ1(x) = 1/2ς(x1)ς(x2),µ0(x) = −1/2ς(x1)ς(x2)
    
    df['Y'] =  -1/2*sig(df['X1'])*sig(df['X2']) + df['Z'] * df['tau']  + df['epsilon']
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

def Causal_SLearner(data, models):

    X = data[[col for col in data.columns if col.startswith('X')]]
    T = data['Z'] #treatment indicator
    y = data['Y']
    est = SLearner(overall_model=models)
    est.fit(y, T, X=X)
    tau_hat_s = est.effect(X)
    return tau_hat_s

def Causal_TLearner(data, models):
   
    X = data[[col for col in data.columns if col.startswith('X')]]
    T = data['Z'] #treatment indicator
    y = data['Y']
    est = TLearner(models=models)
    est.fit(y, T, X=X)
    tau_hat_t = est.effect(X)
    return tau_hat_t

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
