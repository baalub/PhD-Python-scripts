import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

import scipy.integrate as integrate
from matplotlib import pyplot
from numpy.linalg import inv

# Truncated and non-truncated densities on [a,b]

def p(m,s,x):
    """
    Probability density function of the non-truncated Gumbel distributions
    inputs:
    m (float): mode
    s (float): the scale
    x (float): evaluation point of the pdf 

    output (float): the value of the pdf with mode m and scale s at point x
    """  
    y = (x-m)/s
    return  1/s * np.exp( -y -np.exp(-y))

    
def F(x):
    """
    Cumulative distribution function of (0,1) Gumbel density
    """
    return np.exp(-np.exp(-x))

def p_prime(x):
    """
    Derivative of the non-truncated (0,1) Gumbel density
    """
    return (-1+np.exp(-x))*p(0,1,x)


def q(m,s,x,a,b): 
    """
    Truncated Gumbel pdf
    m (float): mode
    s (float): the scale
    x (float): evaluation point of the pdf
    a,b (floats): truncation bounds
    
    output (float): value of the truncated pdf on [a,b] with parameter (m,s) on point x 
    """
    y = (x-m)/s
    if x>b or x<a:
        return 0
    else:
        y_b = (b-m)/s
        y_a = (a-m)/s
        P_theta = F(y_b) - F(y_a)  # this is the normalization constant
        return p(m,s,x)/P_theta


def q_trunc_inputs(m,s,X,a,b):
    """
    Parallelized version of a truncated Gumbel pdf
    m (float): mode
    s (float): the scale
    x (float): evaluation point of the pdf
    a,b (floats): truncation bounds
    
    output (float): list of the truncated pdf on [a,b] with parameter (m,s) evaluated on points in X
    """
    y_b = (b-m)/s
    y_a = (a-m)/s
    
    P_theta = F(y_b) - F(y_a)
    return p(m,s,X)/P_theta

def N(m,s,a,b):
    """Normalization constant
    m (float): mode
    s (float): scale
    a,b (floats): truncation bounds

    output (float): the Normalization constant for the [a,b]-truncated Gumbel density with parameter (m,s)
    """
    y_b = (b-m)/s
    y_a = (a-m)/s
    return F(y_b) - F(y_a) 
    
    
def grad_Hess_logN(m,s,a,b):
    """Helper function for computing the partial derivatives of the normalization constant"""
    grad_logN=np.zeros(2)
    Hess_logN=np.zeros((2,2))
    
    b_theta = (b-m)/s
    a_theta = (a-m)/s
    
    partial_m_a_theta, partial_m_b_theta = -1/s,-1/s
    partial_s_a_theta, partial_s_b_theta = -a_theta/s, -b_theta/s    
    partial_mm_a_theta, partial_mm_b_theta = 0, 0
    partial_ms_a_theta, partial_ms_b_theta = 1/s**2, 1/s**2
    partial_ss_a_theta, partial_ss_b_theta = 2*a_theta/s**2, 2*b_theta/s**2
    
    Nmsab = N(m,s,a,b)
    
    pmsb = p(m,s,b)
    pmsa = p(m,s,a)
    p01b_theta = p(0,1,b_theta)
    p01a_theta = p(0,1,a_theta)
    
    #grad logN
    grad_logN[0] = (-pmsb + pmsa)/Nmsab
    grad_logN[1] = (-pmsb*b_theta + pmsa*a_theta)/Nmsab
    
    #Hess logN
        #Hess12
    A1ms = p_prime(b_theta)* partial_m_b_theta * partial_s_b_theta + p01b_theta*partial_ms_b_theta 
    A2ms = p_prime(a_theta)* partial_m_a_theta * partial_s_a_theta + p01a_theta*partial_ms_a_theta
    
    Bms = p01b_theta*partial_s_b_theta - p01a_theta*partial_s_a_theta
    
    partial_m_N = -pmsb + pmsa
    
    Hess_logN[0,1] = ((A1ms-A2ms)*Nmsab - partial_m_N*Bms)/Nmsab**2
    Hess_logN[1,0] = Hess_logN[0,1]
    
        #Hess11
    A1mm = p_prime(b_theta)* partial_m_b_theta**2 + p01b_theta*partial_mm_b_theta 
    A2mm = p_prime(a_theta)* partial_m_a_theta**2 + p01a_theta*partial_mm_a_theta
    
    Bmm = p01b_theta*partial_m_b_theta - p01a_theta*partial_m_a_theta
    
    partial_m_N = p01b_theta*(-1/s) - p01a_theta*(-1/s)
    
    Hess_logN[0,0] = ((A1mm-A2mm)*Nmsab - partial_m_N*Bmm)/Nmsab**2
        
        #Hess22
    A1ss = p_prime(b_theta)* partial_s_b_theta**2 + p01b_theta*partial_ss_b_theta 
    A2ss = p_prime(a_theta)* partial_s_a_theta**2 + p01a_theta*partial_ss_a_theta
    
    Bss = p01b_theta*partial_s_b_theta - p01a_theta*partial_s_a_theta
    
    partial_s_N = p01b_theta*(-b_theta)/s - p01a_theta*(-a_theta/s)
    
    Hess_logN[1,1] = ((A1ss-A2ss)*Nmsab - partial_s_N*Bss)/Nmsab**2
    
    return grad_logN,Hess_logN
    
    
# FIM in the non-truncated case

gam = 0.57721566

def func2(x):
    return np.log(x)**2 *x* np.exp(-x)

def fun3(x):
    return -np.log(x)*np.exp(-x)*x

def fun4(x):
    return np.exp(-2*x-np.exp(-x))*x**2


gam_der = integrate.quad(func2,0.001,1000)
gam_der_2 = integrate.quad(fun3,0.0001,1000)
gam_der_3 = integrate.quad(fun4,-100,100)

def I(m,s):
    """
    Fisher information for the non-truncated Gumbel family
    m (float): parameter
    s (float): parameter
    
    output (np.ndarray): FIM of the Gumbel family at point (m,s)"""

    I_11 = 1/s**2
    I_12 = (gam-1)/s**2
    I_22 = (gam_der[0]+1)/s**2
    
    return np.array([[I_11,I_12],
                     [I_12,I_22]])


# FIM in the truncated case

def fun(y):
    return y*p(0,1,y)
    
def fun2(y):
    return y**2*np.exp(-2*y - np.exp(-y))

def fun1(y):
    return y*p(0,1,y)
    

def J(m,s,a,b):
    """
    Fisher information for the truncated Gumbel family

    m (float): parameter
    s (float): parameter
    
    output (np.ndarray): FIM of the [a,b]-truncated Gumbel family at point (m,s)"""

    a_theta = (a-m)/s
    b_theta = (b-m)/s
    
    grad,Hess = grad_Hess_logN(m,s,a,b)
    Nmsab = N(m,s,a,b)
    
    
    EX = integrate.quad(fun, a_theta,b_theta)[0]/Nmsab
    EeX = 1 - s*grad[0]
    EXeX  = -1 - s*grad[1] + EX 
    EX2eX = integrate.quad(fun2,a_theta,b_theta)[0]/Nmsab 

    J_11 = EeX/s**2 + Hess[0,0] # vérifié
    J_12 = 1/s**2 - EeX/s**2 + EXeX/s**2 + Hess[0,1]  # vérifié
    J_22 = 1/s**2 * ( 1 + 2*s*grad[1] + EX2eX) + Hess[1,1] # vérifié

    return np.array([[J_11,J_12],
                     [J_12, J_22]])


# partial derivatives of the FIM

def partial_J(typ,m,s,a,b):
    """Helper function for computing the partial derivatives of the FIM"""
    eps = 1.e-7

    if typ == "m":
        return (J(m + eps,s,a,b) - J(m,s,a,b))/eps
    
    if typ == "s":
        return (J(m,s+eps,a,b) - J(m,s,a,b))/(eps)

# Approximation of Christoffel symbols

def ind(i):
    if i==1:
        return "m"
    if i==2:
        return "s"
    
def J_inv(m,s,a,b):
    """Inverse Fisher information matrix"""
    J_inverse = np.linalg.inv(J(m,s,a,b))
    return J_inverse

def coef(i,s):
    if i==1:
        return 0
    else:
        return -2/s**3

A = np.array([[1,gam -1],[gam -1, gam_der[0]+1]])
det_A = np.linalg.det(A)
B = np.array([[gam_der[0]+1, 1-gam],[1-gam, 1]])  # matrix transpose of the comatrix of A

def Gamma_Gum_non_tr(i,j,k,m,s):
    """
    Christoffel symbols of the non-truncated Gumbel family 

    i (int): Christoffel symbol coefficient
    j (int): Christoffel symbol coefficient
    k (int): Christoffel symbol coefficient
    m (float): parameter
    s (float): parameter
    
    output (float): Christoffel symbols in the non-truncated case with coefficient (i,j,k) at point (m,s)"""

    L = [ B[k-1,l-1]*s**2/det_A*(  coef(j,s)*A[l-1,i-1] +coef(i,s)*A[l-1,k-1] -coef(l,s)*A[i-1,j-1] ) for l in [1,2]]
    return np.sum(L)

def Gamma_Gum(i,j,k,m,s,a,b):
    """
    Christoffel symbols of the non-truncated Gumbel family 

    i (int): Christoffel symbol coefficient
    j (int): Christoffel symbol coefficient
    k (int): Christoffel symbol coefficient
    m (float): parameter
    s (float): parameter
    
    output (float): Christoffel symbols in the truncated case with coefficient (i,j,k) at point (m,s)"""

    J_inverse = J_inv(m,s,a,b)
    L = [0.5*J_inverse[k-1,l-1]*(partial_J(ind(j),m,s,a,b)[l-1,i-1] + partial_J(ind(i),m,s,a,b)[l-1,k-1] - partial_J(ind(l),m,s,a,b)[i-1,j-1]) for l in [1,2]]
    return np.sum(L)
    
    

def H_Gum(m,s,dm,ds,a,b):
    """
    Helper function corresponding to the vector field defining the geodesic equation 
    in the truncated case
    
    m (float): parameter point
    s (float): parameter point
    dm (float): first coordinate of direction at point (m,s)
    ds (float): second coordinate of direction at point (m,s)
    a,b: truncation bounds

    output (np.ndarray): Value of the vector field at point (m,s,dm,ds)
    """

    A = -Gamma_Gum(1,1,1,m,s,a,b)*dm**2 -2*Gamma_Gum(1,2,1,m,s,a,b)*dm*ds -Gamma_Gum(2,2,1,m,s,a,b)*ds**2
    B = -Gamma_Gum(1,1,2,m,s,a,b)*dm**2 -2*Gamma_Gum(1,2,2,m,s,a,b)*dm*ds -Gamma_Gum(2,2,2,m,s,a,b)*ds**2

    return np.array([dm,ds,A,B])

def H_Gum_non_tr(m,s,dm,ds):
    """
    Helper function corresponding to the vector field defining the geodesic equation
    in the non truncated case

    m (float): parameter point
    s (float): parameter point
    dm (float): first coordinate of direction at point (m,s)
    ds (float): second coordinate of direction at point (m,s)
    a,b: truncation bounds

    output (np.ndarray): Value of the vector field at point (m,s,dm,ds)
    """

    A = -Gamma_Gum_non_tr(1,1,1,m,s)*dm**2 -2*Gamma_Gum_non_tr(1,2,1,m,s)*dm*ds -Gamma_Gum_non_tr(2,2,1,m,s)*ds**2
    B = -Gamma_Gum_non_tr(1,1,2,m,s)*dm**2 -2*Gamma_Gum_non_tr(1,2,2,m,s)*dm*ds -Gamma_Gum_non_tr(2,2,2,m,s)*ds**2

    return np.array([dm,ds,A,B])



def geod_Gumb_tronquees(Tf,h,Y_0,a,b):
    """
    ODE solver (Euler method) for the geodesic equation in the truncated case
    
    Tf (float): Corresponds to the endpoint of the interval on which the equation is solved i.e. [0,Tf]
    h (float): step size for Euler method
    X_0 (array): initial conditions
    a,b (floats): truncation bounds

    output (tuple): a tuple of two arrays corresponding to the x and y coordinates of the approximated geodesic
    """

    N = np.int64(Tf/h)
    Y=np.zeros((N,4))
    Y[0,:] = Y_0
    
    for n in range(N-1):
        Y[n+1,:]= Y[n,:] + h*H_Gum(Y[n,0],Y[n,1],Y[n,2],Y[n,3],a,b)
        
    return Y[:,0],Y[:,1]

def geod_Gumb_non_tronquees(Tf,h,Y_0):
    """ODE solver (Euler method) for the geodesic equation in the non-truncated case

    Tf (float): Corresponds to the endpoint of the interval on which the equation is solved i.e. [0,Tf]
    h (float): step size for Euler method
    X_0 (array): initial conditions

    output (tuple): a tuple of two arrays corresponding to the x and y coordinates of the approximated geodesic
    """

    N = np.int64(Tf/h)
    Y=np.zeros((N,4))
    Y[0,:] = Y_0
    
    for n in range(N-1):
        Y[n+1,:]= Y[n,:] + h*H_Gum_non_tr(Y[n,0],Y[n,1],Y[n,2],Y[n,3])
        
    return Y[:,0],Y[:,1]
    
# Discretization of geodesics spheres in the truncated case.
# For each geodesic, it solves the geodesic equation with Euler method. 


def tracer_sphere_tr_avec_geod_Gumb(p,delta,nb_pts,h,a,b):
    """Helper function for discretizing the geodesic sphere in the truncated case

    p (array): coordinates of the center of the sphere
    delta (float): radius of the sphere
    nb_pts (int): number of discretization points on the sphere
    h (float): time step for solving the geodesic equation
    a,b (floats): truncation bounds
    
    outputs: 
    n (int): number of time steps of size h when solving each geodesic equation
    C (np.ndarray): list of size nb_pts containing arrays of size 1x2 that are endpoints of the approximated geodesics
    Liste (np.ndarray): array (list) of arrays containing the x and y coordinates of each approximated geodesic"""

    m=p[0]
    s=p[1]
    L = np.linspace(0,2*np.pi,nb_pts)
    j = 0    
    n = np.int64(1/h)
    C = np.zeros((nb_pts,2))
    Liste = []
    for t in L:
        v = np.array([np.cos(t),np.sin(t)])
        l_J = np.sqrt(np.dot(np.transpose(v),np.dot(J(m,s,a,b),v)))
        v_J = delta*(v/l_J)
        X0_J = np.array([m,s,v_J[0],v_J[1]])
        X,Y = geod_Gumb_tronquees(1,h,X0_J,a,b)
        Liste.append([X,Y])
        C[j,:] = np.array([ X[-1],Y[-1] ])
        j+=1

    return n,C,np.array(Liste)

def tracer_sphere_nontr_avec_geod(p,delta,nb_pts,h):
    """Helper function for discretizing the geodesic sphere in the non-truncated case

    p (array): coordinates of the center of the sphere
    delta (float): radius of the sphere
    nb_pts (int): number of discretization points on the sphere
    h (float): time step for solving the geodesic equation
    
    outputs: 
    n (int): number of time steps of size h when solving each geodesic equation
    C (list): list of size nb_pts containing arrays of size 1x2 that are endpoints of the approximated geodesics
    Liste (np.ndarray): array (list) of arrays containing the x and y coordinates of each approximated geodesic"""

    p1=p[0]
    p2=p[1]
    #nb_pts = 
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
        X,Y = geod_Gumb_non_tronquees(1,h,X0_J)
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
    
    th (np.array): array of size 1x2 of the target parameter for the importance sampling procedure
    th0 (np.ndarray): array of size 1x2 of the initial parameter for the importance sampling procedure
    X (np.ndarray): sample point(s) on which the likelihood ratio will be evaluated
    a,b (float): truncation bounds
    
    output (np.ndarray): float or np.ndarray likelihood ratio(s) of the pdf with parameter th over the pdf with parameter th0"""

    m,s = th[0],th[1]
    m0,s0 = th0[0],th0[1]
    
    q_th = q_trunc_inputs(m,s,X,a,b) 
    q_th_0 = q_trunc_inputs(m0,s0,X,a,b)
    
    return q_th/q_th_0


def F_is(sample,H_val,theta_0,theta,t,a,b):
    """ 
    Computes the empirical cdf. This function is taken from the online supplementary code 
    of the paper [Gauchy et al., 2022] in Technometrics.
    
    sample: sample from the truncated Gumbel variable with parameter theta_0
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
    M = np.array([L, ] * len(t))    # duplicate the list containing the likelihood ratios
    B = M * Bool                    # compute the product under the summation
    
    return np.sum(B, axis=1) / np.sum(L)


def quant_estim(sample,H_val,theta_0,theta,alpha,a,b):
    """
    Helper function for computing the quantile estimator. This function is taken from the online supplementary code 
    of the paper [Gauchy et al., 2022] in Technometrics.

    sample: sample from the truncated Gumbel variable with parameter theta_0
    H_val: value of the model G on the sample
    theta_0: initial parameter
    theta: target parameter for the importance sampling procedure 
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
    Sample_input (np.ndarray): sample from the truncated Gumbel variable with parameter theta_0
    H_val (np.ndarray): value of the model G on the sample
    theta_0 (np.ndarray): initial parameter
    theta (np.ndarray): target parameter for the importance sampling procedure 
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
    

def sample_nontr(m,s,N):
    """
    Helper function for sampling from a non-truncated Gumbel distribution.
    m (float): mode
    s (float): scale
    N (int): sample size
    
    output (np.ndarray): A sample of size N from the non-truncated Gumbel distribution with parameter (m,s)"""

    U = scs.uniform().rvs(N)
    return m-s*np.log(-np.log(U))
    
    
def sample(m,s,N,a,b):
    """
    Helper function for sampling from a truncated Gumbel distribution
    m (float): mode
    s (float): scale
    a,b (float): truncation interval
    N (int): sample size
    
    output (np.ndarray): A sample of size N from the truncated Gumbel distribution on [a,b] with parameter (m,s)"""
    
    S= np.zeros(N)
    for i in range(N):
        k = sample_nontr(m,s,1)
        while k<=a or k>=b:
            k = sample_nontr(m,s,1)
        S[i] = k
    return S

