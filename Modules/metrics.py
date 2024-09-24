import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def calc_r2(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
    
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
    r2 = calc_r2(tau, tau_hat)

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

def bar_plot(ax, means, errs, xnames, ylabel, title):
    means = means
    errors = errs
    categories = xnames
    
    bars = ax.bar(xnames, means, yerr=errs, capsize=5, color='red', edgecolor='black', alpha=0.7)
    
  
    #ax.bar(categories, means, yerr=errors, capsize=5, color='red', edgecolor='black', alpha=0.7)
    
    # Adding labels and title
    ax.set_xlabel('Learners')
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for bar, mean, error in zip(bars, means, errs):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # X-coordinate (center of the bar)
            height + error + 0.02 * max(means),  # Y-coordinate (above the bar and error bar)
            f'{mean:.2f}\n±{error:.2f}',  # Text displaying the mean and SD
            ha='center',  # Center the text horizontally
            va='bottom',  # Align text to the bottom of the bar
            fontsize=10,  # Font size
            color='black'  # Text color
        )
   