import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

import scipy.integrate as integrate
from matplotlib import pyplot
from numpy.linalg import inv


# Triangular density functions on [a,b]

def q(m,x,a,b):
    """
    Probability density function of the triangular distribution
    inputs:
    m (float): mode
    x (np.ndarray): evaluation point(s) of the pdf 
    a,b (float): truncation bounds

    output (float): the value of the triangular pdf with support [a,b] and mode m at point x
    """

    A = np.ones(len(x))*a
    B = np.ones(len(x))*b
    bool1 = (x<=m) & (x>a)
    bool2 = (x>m) & (x<b)
    num = (x-A)/(m-a) * bool1 + (B-x)/(b-m) * bool2 
    den = (b-a)
    return 2*num/den
        
#def F(m,x,a,b):
#    if x < a:
    #     return 0
    # if x >b:
    #     return 1
    # if x >= a and x<=m:
    #     return (x-a)**2/( (b-a)*(m-a) )
    # if x > m and x <b:
    #     return 1 - (b-x)**2/( (b-a)*(b-m) )  

def J(m,a,b):
    """
    Helper function for computing the Fisher information for the triangular family

    m (float): mode
    a,b (float): truncation bounds
    
    output (float): the Fisher information at point m in the triangular family supported on [a,b]
    """

    return 1/(b-a) * ( 1/(b-m) +  1/(m-a)  )

#def J_prime(m,a,b):
#    return 1/(b-a) * ( 1/((b-m)**2) +  1/((m-a)**2) )
    
def spheres_exact(center,delta,a,b):
    """
    Helper function for computing the exact Fisher-Rao sphere points in the triangular family
    
    center (float): center of the sphere
    delta (float): radius of the sphere
    a,b (float): truncation bounds 

    output (list): a list of two points corresponding to the sphere points
    """
    alpha = (a+b)/2
    beta = (a-b)**2/4
    m_al_bet = (center - alpha)/np.sqrt(beta)
    
    s_plus = np.sqrt(beta)*np.sin( delta + np.arcsin(m_al_bet  ) ) + alpha
    s_moins = np.sqrt(beta)*np.sin( np.arcsin(m_al_bet) - delta ) + alpha
    
    return [s_moins,s_plus]
    
    
#=================================================================================
#========================== Statistical functions ================================
#=================================================================================


# Likelihood ratios of triangular densities
     
def likelihood_ratio(m,m0,X,a,b):
    """
    Helper function for computing the likelihood ratios of two pdfs truncated on [a,b]
    
    m (float): the target parameter for the importance sampling procedure
    m0 (float): the initial parameter for the importance sampling procedure
    X (np.ndarray): sample point(s) on which the likelihood ratio will be evaluated
    a,b (float): truncation bounds
    
    output (np.ndarray): likelihood ratio(s) of the pdf with parameter th over the pdf with parameter th0"""

    q_th = q(m,X,a,b) 
    q_th_0 = q(m0,X,a,b)
    
    return q_th/q_th_0


def F_is(sample,H_val,m0,m,t,a,b):
    """ 
    Computes the empirical cdf. This function iq taken from the online supplementary code 
    of the paper [Gauchy et al., 2022] in Technometrics.

    sample (np.ndarray): sample from the triangular density with parameter theta_0
    H_val (np.ndarray): values of the model G on the sample
    m (float): target parameter for the importance sampling procedure
    m_0 (float): initial parameter
    t (np.ndarray): evaluation point(s) for the empirical cdf
    
    output (np.ndarray) : the value(s) of the empirical cdf built using sample and evaluated on t
    """
    
    N=len(sample)
    L = likelihood_ratio(m,m0,sample,a,b)  
    G = np.array([H_val, ] * len(t))  
    T = np.array([t, ] * len(sample))  

    Bool = G <= T.transpose()       

    M = np.array([L, ] * len(t))     
    B = M * Bool                     
    
    return np.sum(B, axis=1) / np.sum(L)


def quant_estim(sample,H_val,m0,m,alpha,a,b):
    """
    Helper function for computing the quantile estimator. This function is taken from the online supplementary code 
    of the paper [Gauchy et al., 2022] in Technometrics.

    sample (np.ndarray): sample from the truncated normal variable with parameter theta_0
    H_val (np.ndarray): value of the model G on the sample 
    m0 (float): initial parameter 
    m (float): target parameter 
    alpha (float): quantile order

    output (float): empirical quantile
    """
    
    F = np.array([F_is(sample,H_val,m0,m,[k],a,b) for k in H_val])
    
    boolean =  F <= alpha  
    
    return np.sort(H_val)[np.sum(boolean)-1]

def quantile_estim_fast(F_is,Sample_input,H_val,m0,m,alpha,a,b):
    """
    Helper function for computing the quantile estimator. This method is faster than the previous one because
    it does not need to compute the value of the empirical cdf for all the sample points.
    
    F_is (function): the function for computing the empirical cdf 
    Sample_input (np.ndarray): sample from the truncated Gumbel variable with parameter theta_0
    H_val (np.array): value of the model G on the sample
    m0 (float): initial parameter
    theta (float): target parameter for the importance sampling procedure 
    alpha (float): quantile order
    a,b (float): truncation bounds

    output (float): empirical quantile of order alpha
    """

    H_val_ord = np.sort(H_val)
    i = int(alpha*len(H_val))
    
    # if the cdf is smaller than alpha on the ith element of the ordered sample H_val
    # we increment i
    
    if F_is(Sample_input,H_val,m0,m,[H_val_ord[i]],a,b) < alpha:
        while F_is(Sample_input,H_val,m0,m,[H_val_ord[i]],a,b) < alpha:
            i = i+1
        return H_val_ord[i-1] # the -1 is because of python's indexing convention

    # if the cdf is larger than alpha on the ith element of the ordered sample H_val
    # we decrement i
    else:
        while F_is(Sample_input,H_val,m0,m,[H_val_ord[i]],a,b)>= alpha:
            i = i-1
        return H_val_ord[i] #again we use i instead of i-1 because of python's indexing convention
    
   
def sample(m,N,a,b):
    """
    Helper function for sampling from a triangular distribution

    m (float): mode
    N (int): sample size
    a,b (float): truncation bounds
    
    output (np.ndarray): A sample of size N from the triangular distribution supported on [a,b] with mode m"""
    U=scs.uniform().rvs(N)
    A = np.ones(N)*a+np.sqrt(U*(b-a)*(m-a)) 
    B = np.ones(N)*b-np.sqrt( (np.ones(N) - U)*(b-a)*(b-m) )
    boolean = U < (m-a)/(b-a)
    
    return A*boolean + B*(1-boolean)
    
    


