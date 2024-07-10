import pandas as pd
import numpy as np

def simulation_1(n, p, beta=1, sigma=1):
    """
    Simulate data for the specified model using pandas.
    
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
