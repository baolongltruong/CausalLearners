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
from scipy.stats import beta


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

def simulation_XLearner_1(n, p=20):
    '''Unbalanced case with a simple CATE'''
    
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

    beta = np.random.uniform(-5, 10, p) 
    mu0 = np.dot(df[[f'X{i+1}' for i in range(p)]], beta) + 5 * np.where(df['X1'] > 0.5, 1, 0)
    
    df['Y'] = mu0 + df['Z'] * df['tau'] + df['epsilon']
    
    return df


def simulation_XLearner_2(n,p=20):
    ''' Balanced case, no confounding, with complex non-linear CATE'''
    def _sig(x):
        return 2 / (1 + np.exp(-12 * (x - 1/2)))
        
    X = np.random.normal(0, 1, (n, p))
    
    # Create DataFrame with covariates
    df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(p)])
    
    # µ1(x) = 1/2ς(x1)ς(x2),µ0(x) = −1/2ς(x1)ς(x2)
    df['tau'] = _sig(df['X1'])*_sig(df['X2'])

    
    df['Z'] = np.random.binomial(1, 0.5, n)
    df['epsilon'] = np.random.normal(0, 1, n)


    # µ1(x) = 1/2ς(x1)ς(x2),µ0(x) = −1/2ς(x1)ς(x2)
    
    df['Y'] =  -1/2*_sig(df['X1'])*_sig(df['X2']) + df['Z'] * df['tau']  + df['epsilon']
    
    return df

def simulation_XLearner_3(n,p=20):
    ''' Balanced case, no confounding, with complex linear CATE'''
    X = np.random.normal(0, 1, (n, p))

    # Create DataFrame with covariates
    df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(p)])
    
    #µ1(x) = xTβ1, with β1 ∼ Unif([1, 30]20),µ0(x) = xT β0, with β0 ∼ Unif([1, 30]20).

    mu1 = np.dot(df[[f'X{i+1}' for i in range(p)]], np.random.uniform(1, 30, p))  # µ1(x) = x^T * β1
    mu0 = np.dot(df[[f'X{i+1}' for i in range(p)]], np.random.uniform(1, 30, p))  # µ0(x) = x^T * β0

    df['tau'] = mu1 - mu0
    
    df['Z'] = np.random.binomial(1, 0.5, n)
    df['epsilon'] = np.random.normal(0, 1, n)

    df['Y'] = mu0 + df['Z'] * df['tau'] + df['epsilon']
    
    return df

def simulation_XLearner_4(n,p=5):
    '''No Treatment effect, global linear CATE'''
    X = np.random.normal(0, 1, (n, p))

    # Create DataFrame with covariates
    df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(p)])

 
    mu0 = np.dot(df[[f'X{i+1}' for i in range(p)]], np.random.uniform(1, 30, p))  # µ0(x) = x^T * β0

    df['tau'] = 0
    
    df['Z'] = np.random.binomial(1, 0.5, n)
    df['epsilon'] = np.random.normal(0, 1, n)

    df['Y'] = mu0 + df['Z'] * df['tau'] + df['epsilon']
    return df


def simulation_XLearner_5(n, p=20):
    '''Beta Confounded Case with non-constant propensity score '''
    X = np.random.uniform(0, 1, (n, p))
    
    # Create DataFrame with covariates
    df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(p)])
    
    # μ0(x) = 2x1 − 1, μ1(x) = μ0(x).
    mu0 = 2 * df['X1'] - 1
  
    
    # Define propensity score e(x) with Beta(2, 4) distribution for confounding
    beta_vals = beta.rvs(2, 4, size=n)
    df['e_x'] = 1 / 4 * (1 + beta_vals)  # e(x) = 1/4 * (1 + β(x1,2,4))
    
    df['tau'] = 0
    
    df['Z'] = np.random.binomial(1, df['e_x'])
    df['epsilon'] = np.random.normal(0, 1, n)
    
    df['Y'] = mu0 + df['Z'] * df['tau'] + df['epsilon']
    
    return df
    
def Causal_LR_old(data):
    lr_xfit = data.copy()
    lr_xfit['X1*Z'] = lr_xfit['X1'] * lr_xfit['Z'] #Setting interaction term
    lr_xfit = lr_xfit[['X1', 'Z', 'X1*Z']]
    
    lr = LinearRegression() #Fit linear regression
    lr.fit(lr_xfit, data['Y'])
    
    bz = lr.coef_[1]
    bzx = lr.coef_[2]
    
    tau_hat_lr = bz + bzx*data['X1']
    return tau_hat_lr

def Causal_XLearner_old(data, models):
    
    X = data[[col for col in data.columns if col.startswith('X')]]
    T = data['Z'] #treatment indicator
    y = data['Y']
    est = XLearner(models=models)
    est.fit(y, T, X=X)
    tau_hat_x = est.effect(X)
    return tau_hat_x

def Causal_SLearner_old(data, models):

    X = data[[col for col in data.columns if col.startswith('X')]]
    T = data['Z'] #treatment indicator
    y = data['Y']
    est = SLearner(overall_model=models)
    est.fit(y, T, X=X)
    tau_hat_s = est.effect(X)
    return tau_hat_s

def Causal_TLearner_old(data, models):
   
    X = data[[col for col in data.columns if col.startswith('X')]]
    T = data['Z'] #treatment indicator
    y = data['Y']
    est = TLearner(models=models)
    est.fit(y, T, X=X)
    tau_hat_t = est.effect(X)
    return tau_hat_t

def Causal_DRLearner_old(data):

    X = data[[col for col in data.columns if col.startswith('X')]]
    T = data['Z'] #treatment indicator
    y = data['Y']
    est = DRLearner()
    est.fit(y, T, X=X, W=None)
    tau_hat_dr = est.effect(X)
    return tau_hat_dr

def Causal_CausalForest_old(data):
    
    X = data[[col for col in data.columns if col.startswith('X')]]
    T = data['Z'] #treatment indicator
    y = data['Y']
    est = CausalForestDML(discrete_treatment=True)
    est.fit(y, T, X=X, W=None)
    tau_hat_cf = est.effect(X)

    return tau_hat_cf

def Causal_LR(train_data, test_data):
    lr_xfit = train_data.copy()
    lr_xfit['X1*Z'] = lr_xfit['X1'] * lr_xfit['Z']  # Setting interaction term
    lr_xfit = lr_xfit[['X1', 'Z', 'X1*Z']]
    
    lr = LinearRegression()
    lr.fit(lr_xfit, train_data['Y'])
    
    test_xfit = test_data.copy()
    test_xfit['X1*Z'] = test_xfit['X1'] * test_xfit['Z']
    test_xfit = test_xfit[['X1', 'Z', 'X1*Z']]
    
    bz = lr.coef_[1]
    bzx = lr.coef_[2]
    
    tau_hat_lr = bz + bzx * test_data['X1']
    return tau_hat_lr

def Causal_XLearner(train_data, test_data, models):
    X_train = train_data[[col for col in train_data.columns if col.startswith('X')]]
    #print(train_data['Z'].value_counts())
    T_train = train_data['Z']
    y_train = train_data['Y']
    
    X_test = test_data[[col for col in test_data.columns if col.startswith('X')]]
    
    est = XLearner(models=models)
    est.fit(y_train, T_train, X=X_train)
    tau_hat_x = est.effect(X_test)
    return tau_hat_x

def Causal_SLearner(train_data, test_data, models):
    X_train = train_data[[col for col in train_data.columns if col.startswith('X')]]
    T_train = train_data['Z']
    y_train = train_data['Y']
    
    X_test = test_data[[col for col in test_data.columns if col.startswith('X')]]
    
    est = SLearner(overall_model=models)
    est.fit(y_train, T_train, X=X_train)
    tau_hat_s = est.effect(X_test)
    return tau_hat_s

def Causal_TLearner(train_data, test_data, models):
    X_train = train_data[[col for col in train_data.columns if col.startswith('X')]]
    T_train = train_data['Z']
    y_train = train_data['Y']
    
    X_test = test_data[[col for col in test_data.columns if col.startswith('X')]]
    
    est = TLearner(models=models)
    est.fit(y_train, T_train, X=X_train)
    tau_hat_t = est.effect(X_test)
    return tau_hat_t

def Causal_DRLearner(train_data, test_data):
    X_train = train_data[[col for col in train_data.columns if col.startswith('X')]]
    T_train = train_data['Z']
    y_train = train_data['Y']
    
    X_test = test_data[[col for col in test_data.columns if col.startswith('X')]]
    
    est = DRLearner()
    est.fit(y_train, T_train, X=X_train, W=None)
    tau_hat_dr = est.effect(X_test)
    return tau_hat_dr

def Causal_CausalForest(train_data, test_data):
    X_train = train_data[[col for col in train_data.columns if col.startswith('X')]]
    T_train = train_data['Z']
    y_train = train_data['Y']
    
    X_test = test_data[[col for col in test_data.columns if col.startswith('X')]]
    
    est = CausalForestDML(discrete_treatment=True)
    est.fit(y_train, T_train, X=X_train, W=None)
    tau_hat_cf = est.effect(X_test)
    return tau_hat_cf