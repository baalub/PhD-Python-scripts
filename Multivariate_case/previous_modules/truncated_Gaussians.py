import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs


from scipy.integrate import odeint
from numpy.linalg import inv

from functools import wraps
import sys

def memoize(func):
    cache = {}

    @wraps(func)
    def wrapper(*args,**kwargs):
        key = str(args) + str(kwargs)

        if key not in cache:
            cache[key] = func(*args, **kwargs)

        return cache[key]
    
    return wrapper

# Truncated and non-truncated densities on [a,b]

def p(m,s,x):
    """
    Probability density function of the non-truncated normal distributions
    inputs:
    m (float): mean
    s (float): standard deviation
    x (float): evaluation point of the pdf 

    output (float): the value of the pdf with mean m and std s at point x
    """ 

    return  1/(np.sqrt(2*np.pi)*s) * np.exp( -0.5*(x-m)**2/s**2  )

@memoize
def q(m,s,x,a,b):
    """
    Probability density function of the truncated normal distributions
    inputs:
    m (float): mean of the corresponding non-truncated pdf
    s (float): standard deviation of the corresponding non-truncated pdf
    x (float): evaluation point of the pdf
    a,b (floats): truncation bounds 

    output (float): the value of the pdf with mean m and std s at point x
    """ 
    if x>b or x<a:
        return 0
    else:
        P_theta = scs.norm(m,s).cdf(b) - scs.norm(m,s).cdf(a)
        
        return p(m,s,x)/P_theta


def log_normal(m,s,x):
    """
    Probability density function of the non-truncated log normal distribution
    m (float): parameter
    s (float): parameter
    x (float): evaluation point of the pdf

    output (float): the value of the pdf with parameters m and s at point x
    """

    return 1/(x*s*np.sqrt(2*np.pi)) * np.exp( -(np.log(x)-m)**2/(2*s**2)  )
    

def log_normal_tr(m,s,x,a,b):
    """
    Probability density function of the truncated log normal distribution
    m (float): parameter
    s (float): parameter
    x (float): evaluation point of the pdf
    a,b (floats): truncation bounds 

    output (float): the value of the pdf with parameters m and s at point x
    """

    mass = scs.norm(0,1).cdf( (np.log(b) - m)/s ) - scs.norm(0,1).cdf( (np.log(a) - m)/s )
    if x>b or x<a:
        return 0
    else:
        return log_normal(m,s,x)/mass


def N(m,s,a,b):
    """Normalization constant
    m (float): mean
    s (float): standard deviation
    a,b (floats): truncation bounds

    output (float): the Normalization constant for the [a,b]-truncated normal density with parameter (m,s)
    """
    return scs.norm(m,s).cdf(b) - scs.norm(m,s).cdf(a)
    

# We define the conditional mean mu_B and conditional variance sigma_B^2 on [a,b]

def mu_B(m,s,a,b):
    """
    Function for computing the conditional mean of a truncated normal pdf on [a,b]
    m (float): mean parameter
    s (float): scale parameter
    a,b (float): truncation bounds 

    output (float): the conditional mean of the truncated normal pdf on [a,b] with parameter (m,s)
    """

    return m - s*s*(q(m,s,b,a,b) - q(m,s,a,a,b))


def sigma_B_carre(m,s,a,b):
    """
    Function for computing the conditional variance of a truncated normal pdf
    m (float): parameter
    s (float): parameter
    a,b (floats): truncation bounds 

    output (float): the conditional variance of the truncated normal pdf on [a,b] with parameter (m,s)
    """
    return s**2 * ( 1 - ((b-m)*q(m,s,b,a,b) - (a-m)*q(m,s,a,a,b))  ) - (m - mu_B(m,s,a,b) )**2

  
  # First partial derivatives of q w.r.t. parameters mu and sigma

@memoize
def grad_q(m,s,x,a,b):
    """
    Function for computing the first partial derivatives of the truncated normal pdf w.r.t. m and s
    m (float): parameter
    s (float): parameter
    x (float): evaluation point of the pdf
    a,b (floats): truncation bounds

    output (np.ndarray):  array of first partial derivatives of the truncated normal pdf on [a,b] evaluated at x with parameter (m,s)
    """

    q_val = q(m,s,x,a,b)
    qmsbab = q(m,s,b,a,b)
    qmsaab = q(m,s,a,a,b)
    
    A = (x-m)/(s**2)  + qmsbab - qmsaab
    B = 1/s  * ( (b-m)*qmsbab  - (a-m)*qmsaab )  - 1/s  +  (x-m)**2/(s**3)
    
    return np.array([q_val*A,q_val*B])


# Second partial derivatives of q w.r.t. parameters mu and sigma

def Hess_q(m,s,x,a,b):
    """
    Function for computing the second partial derivatives of the truncated normal pdf w.r.t. m and s
    m (float): parameter
    s (float): parameter
    x (float): evaluation point of the pdf
    a,b (floats): truncation bounds

    output(tuple): tuple of arrays of second partial derivatives of the truncated normal pdf on [a,b] evaluated at x with parameter (m,s)
    """

    Hess = np.zeros((2,2))
    
    qmsbab = q(m,s,b,a,b)
    qmsaab = q(m,s,a,a,b)
    qmsxab = q(m,s,x,a,b)
    
    grad_q_msxab = grad_q(m,s,x,a,b)
    grad_q_msbab = grad_q(m,s,b,a,b)
    grad_q_msaab = grad_q(m,s,a,a,b)
    
    Amm = (x-m)/(s**2) + qmsbab - qmsaab
    Bmm = -1/(s**2) + grad_q_msbab[0] - grad_q_msaab[0]
    
    Ams = (x-m)/(s**2) + qmsbab - qmsaab
    Bms = -2*(x-m)/(s**3) + grad_q_msbab[1] - grad_q_msaab[1]
    
    Ass = (1/s  * ( (b-m)*qmsbab  - (a-m)*qmsaab )  - 1/s  +  (x-m)**2/(s**3) )
    Bss = (b-m)*qmsbab - (a-m)*qmsaab  
    Css = (b-m)*grad_q_msbab[1] - (a-m)*grad_q_msaab[1]
    
    Hess[0,0] = grad_q_msxab[0]*Amm + qmsxab*Bmm
    Hess[0,1] = grad_q_msxab[1]*Ams + qmsxab*Bms
    Hess[1,0] = Hess[0,1]
    Hess[1,1] = grad_q_msxab[1]*Ass + qmsxab*(-1/(s**2) * Bss + 1/s *Css + 1/(s**2)  - 3*(x-m)**2/(s**4))
    
    return Hess,qmsaab,qmsbab,grad_q_msaab,grad_q_msbab


# First partial derivatives of mu_B and sigma_B^2 w.r.t. m and s

def grad_hess_de_mu_sigma(m,s,a,b):
    """
    Function for computing the first and second partial derivatives of the truncated normal pdf w.r.t. m and s
    m (float): parameter
    s (float): parameter
    a,b (floats): truncation bounds

    output (np.ndarray):  array of first and second partial derivatives of the conditional mean and variance on [a,b] with parameter (m,s)
    """

    grad_mu = np.zeros(2)
    grad_sigma = np.zeros(2)
    
    Hess_mu = np.zeros((2,2))
    Hess_sigma = np.zeros((2,2))
    
    Hess_q_msaab,qmsaab,qmsbab,grad_q_msaab,grad_q_msbab = Hess_q(m,s,a,a,b)
    Hess_q_msbab = Hess_q(m,s,b,a,b)[0]
    
    mu_B_val = mu_B(m,s,a,b)
    
    #grad_mu
    grad_mu[0] = 1 - s**2 * (grad_q_msbab[0] - grad_q_msaab[0] )
    grad_mu[1] = -2*s*(qmsbab - qmsaab)  - s**2 * (grad_q_msbab[1] - grad_q_msaab[1])
    
    #grad_sigma
    Agrad_sigma0 = -qmsbab   + (b-m)*grad_q_msbab[0] + qmsaab - (a-m)*grad_q_msaab[0]
    Bgrad_sigma0 = 1 - grad_mu[0]
    grad_sigma[0] = -s**2 * Agrad_sigma0 - 2*(m - mu_B_val)*Bgrad_sigma0
    
    Agrad_sigma1 = 1 - ( (b-m)*qmsbab - (a-m)*qmsaab )
    Bgrad_sigma1 = (b-m)*grad_q_msbab[1] - (a-m)*grad_q_msaab[1]
    grad_sigma[1] = 2*s*Agrad_sigma1 - s**2*Bgrad_sigma1 + 2*(m - mu_B_val) * grad_mu[1]
    
    #Hess_mu
    Hess_mu[0,0] = -s**2 * (Hess_q_msbab[0,0]- Hess_q_msaab[0,0])
    Hess_mu[0,1] = -2*s*(grad_q_msbab[0] - grad_q_msaab[0]) -s**2 * (Hess_q_msbab[0,1] - Hess_q_msaab[0,1])
    Hess_mu[1,0] = Hess_mu[1,0]
    Hess_mu[1,1] = -2*(qmsbab - qmsaab) - 4*s*(grad_q_msbab[1] - grad_q_msaab[1])  - s**2*(Hess_q_msbab[1,1] - Hess_q_msaab[1,1])

    #Hess_sigma
    Ahess_sigma00 =  -2*grad_q_msbab[0]  + (b-m)*Hess_q_msbab[0,0] -( -2*grad_q_msaab[0]  + (a-m)*Hess_q_msaab[0,0])
    Bhess_sigma00 = (1 - grad_mu[0])**2
    Hess_sigma[0,0] = -s**2 *Ahess_sigma00 -2*Bhess_sigma00 +2*(m - mu_B_val)*Hess_mu[0,0] 


    Ahess_sigma01 =  -qmsbab + (b-m)*grad_q_msbab[0] +qmsaab - (a-m)*grad_q_msaab[0]
    Bhess_sigma01 =  -grad_q_msbab[1] + (b-m)*Hess_q_msbab[0,1] - (-grad_q_msaab[1] + (a-m)*Hess_q_msaab[0,1])
    Chess_sigma01 =  2 *grad_mu[1]*(1 - grad_mu[0])
    Dhess_sigma01 =  2 *Hess_mu[0,1]*(m - mu_B_val)
    Hess_sigma[0,1] = -2*s*Ahess_sigma01 - s**2*Bhess_sigma01 + Chess_sigma01 + Dhess_sigma01
    Hess_sigma[1,0] = Hess_sigma[0,1]
    
    Ahess_sigma11 = 1 - ((b-m)*qmsbab - (a-m)*qmsaab)
    Bhess_sigma11 = (b-m)*grad_q_msbab[1] - (a-m)*grad_q_msaab[1] 
    Chess_sigma11 = (b - m)*Hess_q_msbab[1,1] - (a - m)*Hess_q_msaab[1,1]
    Dhess_sigma11 = 2* ( grad_mu[1])**2
    Ehess_sigma11 = 2*(m - mu_B_val)* Hess_mu[1,1] 
    Hess_sigma[1,1] = 2*Ahess_sigma11 - 4*s*Bhess_sigma11 - s**2 * Chess_sigma11 + Dhess_sigma11 - Ehess_sigma11
    
    return grad_mu,grad_sigma,Hess_mu,Hess_sigma


# The FIM in the non-truncated case

def I(m,s): 
    """FIM for the Normal family
    m (float): parameter
    s (float): parameter
    
    output (np.ndarray): array of size 2x2 of the Fisher information matrix at point (m,s)""" 
    
    i = np.zeros((2,2))
    
    i[0,0] = 1/(s**2)
    i[0,1] = 0
    i[1,0] = 0 
    i[1,1] = 2/s**2
    
    return i


 # The FIM in the truncated case
 
def J(m,s,a,b): 
    """FIM for the truncated Normal family
    m (float): parameter
    s (float): parameter
    a,b (floats): truncation bounds

    output (np.ndarray): array of size 2x2 of the Fisher information matrix at point (m,s)""" 
      
    J_B = np.zeros((2,2))
    gradmu,gradsigma,Hess1,Hess2 = grad_hess_de_mu_sigma(m,s,a,b)

    J_B[0,0] = gradmu[0] /(s**2)
    J_B[0,1] = gradmu[1]/(s**2)
    J_B[1,0] = J_B[0,1] 
    J_B[1,1] = (gradsigma[1] + 2*(mu_B(m,s,a,b) - m) *gradmu[1] )/(s**3)
    
    return J_B
    
    

def Gamma_LC_prem(m,s,a,b):
    """Function for computing the Christoffel symbols of the first kind for the truncated normal family
    m (float): parameter
    s (float): parameter
    a,b (floats): truncation bounds
    
    output (np.ndarray): array of size 2x2x2 containing Christoffel symbols of the first kind of the [a,b]-truncated normal family
    evaluated at point (m,s)"""

    Gam = np.zeros((2,2,2))
    mu_B_val = mu_B(m,s,a,b)
    
    grad_mu,grad_sigma,Hess_mu,Hess_sigma = grad_hess_de_mu_sigma(m,s,a,b)
    
    partial_m_de_mu_B_val = grad_mu[0]
    partial_s_de_mu_B_val = grad_mu[1]
    
    partial_m_de_sigma_B_carre_val = grad_sigma[0]
    partial_s_de_sigma_B_carre_val = grad_sigma[1]
    
    partial_mm_2_mu_B_val = Hess_mu[0,0]
    partial_sm_2_mu_B_val = Hess_mu[0,1]
    partial_ss_2_mu_B_val = Hess_mu[1,1]
    
    partial_mm_2_sigma_B_carre_val = Hess_sigma[0,0]
    partial_sm_2_sigma_B_carre_val = Hess_sigma[0,1]
    partial_ss_2_sigma_B_carre_val = Hess_sigma[1,1]
    
    #\Gamma^1
    Gam[0,0,0] = partial_mm_2_mu_B_val/(s**2) /2
    Gam[0,1,0] = (partial_sm_2_mu_B_val/(s**2) -2*partial_m_de_mu_B_val/(s**3) )/2
    Gam[1,0,0] = Gam[0,1,0]
    Gam[1,1,0] = (partial_ss_2_mu_B_val/(s**2) -3/(s**4) * (partial_m_de_sigma_B_carre_val + 2*partial_m_de_mu_B_val * (mu_B_val - m)) )/2
    
    #\Gamma^2
    Gam[0,0,1] = 1/(s**3) * (partial_mm_2_sigma_B_carre_val + 2*( mu_B_val - m)*partial_mm_2_mu_B_val + 2*partial_m_de_mu_B_val**2)/2
    Gam[0,1,1] = (1/(s**3) * (partial_sm_2_sigma_B_carre_val + 2*( mu_B_val - m)*partial_sm_2_mu_B_val + 2*partial_m_de_mu_B_val*partial_s_de_mu_B_val) -2*partial_s_de_mu_B_val/(s**3))/2
    Gam[1,0,1] = Gam[0,1,1]
    Gam[1,1,1] = (1/(s**3) * (partial_ss_2_sigma_B_carre_val + 2*( mu_B_val - m)*partial_ss_2_mu_B_val + 2*partial_s_de_mu_B_val**2) -3/(s**4) * (partial_s_de_sigma_B_carre_val + 2*partial_s_de_mu_B_val * (mu_B_val- m)))/2              

    return Gam
    

def Gamma_LC(m,s,a,b):
    """Function for computing the Christoffel symbols of the second kind for the truncated normal family
    m (float): parameter
    s (float): parameter
    a,b (floats): truncation bounds
    
    output (np.ndarray): array of size 2x2x2 containing Christoffel symbols of the second kind of the [a,b]-truncated normal family
    evaluated at point (m,s)"""

    Gam_LC = np.zeros((2,2,2))    # Gam_LC[l,i,j] = \Gamma_{ij}^l
    J_inv = np.linalg.inv(J(m,s,a,b))
    Gam_prem = Gamma_LC_prem(m,s,a,b)
    
    for i in range(2):
        for j in range(2):
            for l in range(2):
                Gam_LC[i,j,l] = Gam_prem[i,j,0]*J_inv[0,l] + Gam_prem[i,j,1]*J_inv[1,l] 
    return Gam_LC     
    
    

def H(V, t, a, b):
    p1, p2, v1, v2 = V[0], V[1], V[2], V[3]

    """
    Helper function corresponding to the vector field defining the geodesic equation 
    in the truncated case
    
    p1 (float): parameter point
    p2 (float): parameter point
    v1 (float): first coordinate of direction at point (m,s)
    v2 (float): second coordinate of direction at point (m,s)
    a,b: truncation bounds

    output (np.ndarray): Value of the vector field at point (m,s,dm,ds)
    """

    Gam_LC = Gamma_LC(p1,p2,a,b)
    A1 = -Gam_LC[0,0,0]*v1**2 -2*Gam_LC[0,1,0]*v1*v2 -Gam_LC[1,1,0]*v2**2
    B1 = -Gam_LC[0,0,1]*v1**2 -2*Gam_LC[0,1,1]*v1*v2 -Gam_LC[1,1,1]*v2**2
    #A = -Gamma_LC_indice(1,1,1,p1,p2,a,b)*v1**2 -2*Gamma_LC_indice(1,2,1,p1,p2,a,b)*v1*v2 -Gamma_LC_indice(2,2,1,p1,p2,a,b)*v2**2
    #B = -Gamma_LC_indice(1,1,2,p1,p2,a,b)*v1**2 -2*Gamma_LC_indice(1,2,2,p1,p2,a,b)*v1*v2 -Gamma_LC_indice(2,2,2,p1,p2,a,b)*v2**2

    return np.array([v1,v2,A1,B1])
 
def G(V, t):
    x,y,u,v = V[0], V[1], V[2], V[3]
    """Vector field defining the geodesic equation for the non-truncated normal family"""
    return np.array([u,v,2*u*v/y,-u**2/(2*y) + v**2/y])

if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()

    H([1, 2, 3, 4], t = 2, a=-1, b=1)
    current_time = timeit.default_timer()
    print(f"It took {current_time - start} seconds")

    G([1, 2, 3, 4], t = 2)

    end_time = timeit.default_timer()
    print(f"It took {end_time - current_time} seconds") 

def geod_non_tronquees(Tf, h, X_0): 
    """
    ODE solver (Euler method) for the geodesic equation in the non-truncated case
    
    Tf (float): Corresponds to the endpoint of the interval on which the equation is solved i.e. [0,Tf]
    h (float): step size for Euler method
    X_0 (array): initial conditions

    output (tuple): a tuple of two arrays corresponding to the x and y coordinates of the approximated geodesic
    """

    N = np.int64(Tf/h)
    #print("in tr Gauss", type(X_0))
    X = odeint(G, X_0, np.linspace(0, Tf, N, endpoint=True))
    
    # for n in range(N-1):
    #    X[n+1,:] = X[n,:] + h*G(X[n,0],X[n,1],X[n,2],X[n,3])
        
    return X[:,0], X[:,1]

# Geodesic solver in the truncated case (Euler method)

def geod_tronquees(Tf,h,Y_0,a,b):
    """
    ODE solver (Euler method) for the geodesic equation in the truncated case
    
    Tf (float): Corresponds to the endpoint of the interval on which the equation is solved i.e. [0,Tf]
    h (float): step size for Euler method
    X_0 (array): initial conditions
    a,b (floats): truncation bounds

    output (tuple): a tuple of two arrays corresponding to the x and y coordinates of the approximated geodesic
    """

    N = np.int64(Tf/h)
    
    Y = odeint(H, Y_0, np.linspace(0, Tf, N, endpoint=True), args=(a, b))
    # for n in range(N-1):
    #     Y[n+1,:]= Y[n,:] + h*H(Y[n,0],Y[n,1],Y[n,2],Y[n,3],a,b)
        
    return Y[:,0], Y[:,1]
    
    
# Discretization of geodesics spheres in the truncated case.
# For each geodesic, it solves the geodesic equation with Euler method.

def tracer_sphere_tr_avec_geod(p,delta,nb_pts,h,a,b):
    """Helper function for discretizing the geodesic sphere in the truncated case

    p (array): coordinates of the center of the sphere
    delta (float): radius of the sphere
    nb_pts (int): number of discretization points on the sphere
    h (float): time step for solving the geodesic equation
    a,b (floats): truncation bounds
    
    output: 
    n (int): number of time steps of size h when solving each geodesic equation
    C (list): list of size nb_pts containing arrays of size 1x2 that are endpoints of the approximated geodesics
    Liste (np.ndarray): array of arrays containing the x and y coordinates of each approximated geodesic"""

    p1=p[0]
    p2=p[1]
    L = np.linspace(0,2*np.pi,nb_pts)
    j = 0    
    n = np.int64(1/h)    # step size for each geodesic
    C = np.zeros((nb_pts,2))    
    Liste = []
    for t in L:
        v = np.array([np.cos(t),np.sin(t)])
        l_J = np.sqrt(np.dot(np.transpose(v),np.dot(J(p1,p2,a,b),v)))
        v_J = delta*(v/l_J)
        X0_J = np.array([p1,p2,v_J[0],v_J[1]])
        X,Y = geod_tronquees(1,h,X0_J,a,b)
        Liste.append([X,Y])
        C[j,:] = np.array([ X[-1],Y[-1] ])
        j+=1
    
    return n,C,np.array(Liste)
    
def concentric_spheres(Liste_geod,n,nb_spheres):
    """Helper function for building the concentric spheres
    
    Liste_geod (list): Contains the list of geodesics starting from the same point
    n (int): number of steps in the geodesic approximation (it corresponds to Tf/h where h is 
    the step size and Tf is the final time for the geodesic)
    nb_spheres (int): number of spheres
    
    output: array where each element is a collection of discretized points on a concentric sphere"""
    
    liste_de_spheres_pt = []
    for i in range(1,nb_spheres+1):
        k=int(n*i/nb_spheres)-1
        XXk = [X[k] for X in Liste_geod[:,0]]
        YYk = [Y[k] for Y in Liste_geod[:,1]]
        liste_de_spheres_pt.append(np.array([XXk,YYk]))
        
    return np.array(liste_de_spheres_pt)


# The following function gives a discretization of geodesic spheres in the non-truncated case.
# For each geodesic, it solves the geodesic equation with Euler method.

def tracer_sphere_nontr_avec_geod(p,delta,nb_pts,h):
    """Helper function for discretizing the geodesic sphere in the non-truncated case

    p (array): coordinates of the center of the sphere
    delta (float): radius of the sphere
    nb_pts (int): number of discretization points on the sphere
    h (float): time step for solving the geodesic equation
    
    output: 
    n (int): number of time steps of size h when solving each geodesic equation
    C (list): list of size nb_pts containing arrays of size 1x2 that are endpoints of the approximated geodesics
    Liste (np.ndarray): array of arrays containing the x and y coordinates of each approximated geodesic"""

    p1=p[0]
    p2=p[1]

    L = np.linspace(0,2*np.pi,nb_pts)
    j = 0    
    n = np.int64(1/h)    # step size for each geodesic
    C = np.zeros((nb_pts,2))   
    Liste = []
    for t in L:
        v = np.array([np.cos(t),np.sin(t)])
        l_J = np.sqrt(np.dot(np.transpose(v),np.dot(I(p1,p2),v)))
        v_J = delta*(v/l_J)
        X0_J = np.array([p1,p2,v_J[0],v_J[1]])
        X,Y = geod_non_tronquees(1,h,X0_J)
        Liste.append([X,Y])
        C[j,:] = np.array([ X[-1],Y[-1] ])
        j+=1
            
    return n,C,np.array(Liste)


#=================================================================================
#========================== Statistical functions ================================
#=================================================================================


# Likelihood ratio of normal densities
    
def likelihood_ratio(th,th0,X,a,b):
    """Helper function for computing the likelihood ratios of two pdfs truncated on [a,b]
    
    th (np.ndarray): array of size 1x2 of the target parameter for the importance sampling procedure
    th0 (np.ndarray): array of size 1x2 of the initial parameter for the importance sampling procedure
    X (np.ndarray): sample point(s) on which the likelihood ratio will be evaluated
    a,b (float): truncation bounds
    
    output (np.ndarray): likelihood ratio(s) of the pdf with parameter th over the pdf with parameter th0"""

    mu,sigma = th[0],th[1]
    mu0,sigma0 = th0[0],th0[1]
    
    q_th = scs.norm(mu,sigma).pdf(X)/N(mu,sigma,a,b)
    q_th_0 = scs.norm(mu0,sigma0).pdf(X)/N(mu0,sigma0,a,b)
    
    return q_th/q_th_0


def F_is(sample,H_val,theta_0,theta,t,a,b):
    """ 
    Computes the empirical cdf. This function iq taken from the online supplementary code 
    of the paper [Gauchy et al., 2022] in Technometrics.

    sample: sample from the truncated normal variable with parameter theta_0
    H_val: value of the model G on the sample
    theta: target parameter for the importance sampling procedure
    theta_0: initial parameter
    t: evaluation point(s) for the empirical cdf
    
    output (np.ndarray): the value(s) of the empirical cdf built using sample and evaluated on t
    """

    N=len(sample)
    L = likelihood_ratio(theta,theta_0,sample,a,b)  # Evaluate the likelihood ratio on the sample points
    G = np.array([H_val, ] * len(t))  # duplicate the array H_val 
    T = np.array([t, ] * len(sample))  # duplicate the points on which the empirical cdf is evaluated

    Bool = G <= T.transpose()       # how many times does H_val surpass t ?

    M = np.array([L, ] * len(t))     # duplicate the list containing the likelihood ratios
    B = M * Bool                     # compute the product under the summation
    
    return np.sum(B, axis=1) / np.sum(L)


def quant_estim(sample,H_val,theta_0,theta,alpha,a,b):
    """
    Helper function for computing the quantile estimator. This function is taken from the online supplementary code 
    of the paper [Gauchy et al., 2022] in Technometrics.
    sample: sample from the truncated normal variable with parameter theta_0
    H_val: value of the model G on the sample 
    theta: target parameter 
    alpha: quantile order

    output (float): empirical quantile of order alpha
    """
    
    F = np.array([F_is(sample,H_val,theta_0,theta,[k],a,b) for k in H_val])
    boolean =  F <= alpha  
    
    return np.sort(H_val)[np.sum(boolean)-1]

  
def quantile_estim_fast(F_is,Sample_input,H_val,theta_0,theta,alpha,a,b):
    """
    Helper function for computing the quantile estimator. This method is faster than the previous one because
    it does not need to compute the value of the empirical cdf for all the sample points.
    
    F_is (function): the function for computing the empirical cdf 
    Sample_input (np.array): sample from the truncated normal variable with parameter theta_0
    H_val (np.array): value of the model G on the sample
    theta_0 (np.array): initial parameter
    theta (np.array): target parameter for the importance sampling procedure 
    alpha (float): quantile order
    a,b (float): truncation bounds
    
    output (float): empirical quantile of order alpha
    """ 

    H_val_ord = np.sort(H_val)
    i = int(alpha*len(H_val))
    
    # if the cdf is smaller than alpha on the ith element of the ordered sample H_val
    # we increment i
    
    if F_is(Sample_input,H_val,theta_0,theta,[H_val_ord[i]],a,b) < alpha:
        while F_is(Sample_input,H_val,theta_0,theta,[H_val_ord[i]],a,b) < alpha:
            i = i+1
        return H_val_ord[i-1] # the -1 is because of python's indexing convention

    # if the cdf is larger than alpha on the ith element of the ordered sample H_val
    # we decrement i
    else:
        while F_is(Sample_input,H_val,theta_0,theta,[H_val_ord[i]],a,b)>= alpha:
            i = i-1
        return H_val_ord[i] #again we use i instead of i-1 because of python's indexing convention


def sample(m,s,N,a,b):
    """
    Helper function for sampling from a truncated normal distribution

    m (float): mean
    s (float): standard deviation
    N (int): sample size
    a,b (float): truncation bounds
    
    output (np.ndarray): A sample of size N from the truncated normal distribution on [a,b] with parameter (m,s)"""
    
    S = np.zeros(N)
    for i in range(N):
        k = scs.norm(m,s).rvs()
        while k<=a or k>=b:
            k = scs.norm(m,s).rvs()
        S[i] = k
    return S
