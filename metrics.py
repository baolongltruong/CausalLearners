import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def evaluate(tau, tau_hat):
    """
    Evaluate the Mean Squared Error (MSE), bias, and R^2 score between tau and tau_hat.
    
    Parameters:
    tau : numpy array
        Array containing actual values.
    tau_hat : numpy array
        Array containing predicted or estimated values.
        
    Returns:
    mse : float
        Mean Squared Error between tau and tau_hat.
    bias : float
        Bias (mean difference) between tau and tau_hat.
    r2 : float
        Coefficient of determination (R^2) score between tau and tau_hat.
    """
    
    mse = np.mean((tau - tau_hat) ** 2)
    bias = np.mean(tau - tau_hat)
    r2 = r2_score(tau, tau_hat)

    return mse, bias, r2

def tau_plot(tau, tau_hat, title):
    """
    Plot tau vs tau_hat with a scatter plot and an ideal y=x line.
    
    Parameters:
    tau : numpy array
        Array containing actual tau values.
    tau_hat : numpy array
        Array containing predicted tau values.
    title : str
        Title for the plot.
    """
    
    plt.figure(figsize=(6, 4)) 
    plt.scatter(tau, tau_hat, color='blue', label='tau_hat vs tau')
    plt.plot(tau, tau, color='red', linestyle='--', label='y = x')  # y=x
    
    # Labeling and customization
    plt.title(title)
    plt.xlabel('tau')
    plt.ylabel('tau_hat')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    
    plt.show()