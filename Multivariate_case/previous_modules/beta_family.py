import openturns as ot
import numpy as np

from scipy.special import polygamma as pgam
from scipy.special import gamma
from scipy.integrate import odeint


def trigam(x):
    return pgam(1,x)

def quadgam(x):
    return pgam(2,x)

def I(alpha,beta):
    """
    Fisher information for the beta family, see the paper Fisher-Rao geometry of Dirichlet distributions
    page 12, Section 4."""
    
    I = np.zeros((2,2))
    
    I[0,0] = trigam(alpha) - trigam(alpha+beta)
    I[0,1] = -trigam(alpha+beta)
    I[1,0] = -trigam(alpha+beta)
    I[1,1] = trigam(beta) - trigam(alpha+beta)
    
    return I

def partial_I(alpha,beta):
    
    partial_alpha_I = np.array([[quadgam(alpha)-quadgam(alpha+beta), -quadgam(alpha+beta)],
                            [-quadgam(alpha+beta), -quadgam(alpha+beta)]])
    
    partial_beta_I = np.array([[-quadgam(alpha+beta), -quadgam(alpha+beta)],
                            [-quadgam(alpha+beta), quadgam(beta) - quadgam(alpha+beta)]])
    
    return np.array([partial_alpha_I, partial_beta_I])


def Christoffel_symbs(alpha,beta):     # Christoffel_symbs[i,j,k] = \Gamma_{ij}^k
    """Christoffel symbols for the beta family"""
    I_inv = np.linalg.inv(I(alpha,beta))   
    partial_I_val = partial_I(alpha,beta)
    
    Gam = np.zeros((2,2,2))
    
    for k in range(2):
        for j in range(2):
            for i in range(2):
                Gam[i,j,k] = np.sum([0.5*I_inv[k,l] *( partial_I_val[j,l,i] + partial_I_val[i,l,j] - partial_I_val[l,i,j] ) for l in range(2) ])
                
    return Gam


def H_Beta(alpha,beta,dalpha,dbeta):
    """Vector field defining the geodesic equation"""
    Gam = Christoffel_symbs(alpha, beta) 
    
    A = -Gam[0,0,0]*dalpha**2 -2*Gam[0,1,0]*dalpha*dbeta -Gam[1,1,0]*dbeta**2
    B = -Gam[0,0,1]*dalpha**2 -2*Gam[0,1,1]*dalpha*dbeta -Gam[1,1,1]*dbeta**2

    return np.array([dalpha, dbeta, A, B])

def geod_Beta(Tf,h,Y_0):
    """ODE solver for the geodesic equation"""
    N = np.int64(Tf/h)
    Y=np.zeros((N,4))
    Y[0,:] = Y_0
    
    for n in range(N-1):
        Y[n+1,:]= Y[n,:] + h*H_Beta(Y[n,0],Y[n,1],Y[n,2],Y[n,3])
        
    return Y[:,0],Y[:,1]
