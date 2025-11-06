import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

import scipy.integrate as integrate
from matplotlib import pyplot
from numpy.linalg import inv
from scipy.special import gamma, gammainc, digamma


def p(alpha,beta,x):
    """pdf of the Gamma distribution with parameter (alpha,beta) in the non-truncated case"""

    return x**(alpha-1)*(beta**alpha * np.exp(-beta*x))/gamma(alpha)
    


def F(alpha,beta,x):
    """cdf of the Gamma distribution with parameter (alpha,beta) in the non-truncated case"""

    return gammainc(alpha,beta*x)

def p_prime(x):
    """derivative pdf of the Gamma distribution with parameter (0,1)"""

    return (-1+np.exp(-x))*p(0,1,x)

def N(alpha,beta,a,b):
   return F(alpha,beta,b) - F(alpha,beta,a)
    
# This is the Gamma density truncated on [a,b] and renormalized  
  
def q(alpha,beta,x,a,b): 
    if x>b or x<a:
        return 0
    else:
        P_theta =   F(alpha,beta,b) - F(alpha,beta,a)  # this is the normalization constant
        return p(alpha,beta,x)/P_theta

def q_trunc_inputs(m,s,X,a,b):
    y_b = (b-m)/s
    y_a = (a-m)/s
    
    P_theta = F(y_b) - F(y_a)
    return p(m,s,X)/P_theta


# non truncated fisher information


#######################################
######### Non truncated case  #########
#######################################



def digamma_prime(x):
    """First derivative of the digamma function"""
    eps  = 1.e-7
    xp = x + eps
    h =  xp - x
    return (digamma(xp) - digamma(x))/h

def digamma_seconde(x):
    """Second derivative of the digamma function"""
    eps  = 1.e-7
    xp = x + eps
    h =  xp - x
    return (digamma_prime(xp) - digamma_prime(x))/h
    
def I(alpha,beta):
    """Fisher information matrix of the Gamma family in the non-truncated case.
    See paper 'statistical manifolds' by Lauritzen page 41, Section 7.
    """
    I_11 = digamma_prime(alpha)
    I_12 = -1/beta
    I_22 = alpha/(beta**2)
    
    return np.array([[I_11,I_12],
                     [I_12,I_22]])

def partial_I(alpha,beta):
    """Partial derivatives of the FIM in the non-truncated case"""

    eps  = 1.e-7
    partial_alpha = [[digamma_seconde(alpha),0],
                              [0,1/beta**2]]
                              
    partial_beta = [[digamma_prime(alpha),1/beta**2],
                             [1/beta**2,-2*alpha/beta**3]]
    
    return np.array([partial_alpha, partial_beta])
    
    
def Christoffel_symbs(alpha,beta):     # Christoffel_symbs[i,j,k] = \Gamma_{ij}^k
    """Christoffel symbols for the Gamma family in the non-truncated case"""
    I_inv = np.linalg.inv(I(alpha,beta))   
    partial_I_val = partial_I(alpha,beta)
    
    Gam = np.zeros((2,2,2))
    
    for k in range(2):
        for j in range(2):
            for i in range(2):
                Gam[i,j,k] = np.sum([0.5*I_inv[k,l] *( partial_I_val[j,l,i] + partial_I_val[i,l,j] - partial_I_val[l,i,j] ) for l in range(2) ])
    
    return Gam
    
def H_Gamma_non_tr(alpha,beta,da,db):
    """Vector field defining the geodesic equation in the non-truncated case"""
    Gam = Christoffel_symbs(alpha,beta) 
    
    A = -Gam[0,0,0]*da**2 -2*Gam[0,1,0]*da*db -Gam[1,1,0]*db**2
    B = -Gam[0,0,1]*da**2 -2*Gam[0,1,1]*da*db -Gam[1,1,1]*db**2

    return np.array([da,db,A,B])
    
def geod_Gamma_non_tronquees(Tf,h,Y_0):
    """ODE solver for the geodesic equation in the non-truncated case"""
    N = np.int64(Tf/h)
    Y=np.zeros((N,4))
    Y[0,:] = Y_0
    
    for n in range(N-1):
        Y[n+1,:]= Y[n,:] + h*H_Gamma_non_tr(Y[n,0],Y[n,1],Y[n,2],Y[n,3])
        
    return Y[:,0],Y[:,1]
    

def tracer_sphere_nontr_avec_geod(p,delta,nb_pts,h):
    """Computes multiple geodesics going from the same starting point"""
    p1=p[0]
    p2=p[1]
    #nb_pts = 
    L = np.linspace(0,2*np.pi,nb_pts)
    j = 0    
    n = np.int64(1/h)    # nombre de pas pour chaque géodésique
    C = np.zeros((nb_pts,2))    # point final de chaque géodésique
    Liste = []
    for t in L:
        v = np.array([np.cos(t),np.sin(t)])
        l = np.sqrt(np.dot(np.transpose(v),np.dot(I(p1,p2),v)))
        v = delta*(v/l)
        
        X0 = np.array([p1,p2,v[0],v[1]])
        
        X,Y = geod_Gamma_non_tronquees(1,h,X0)

        Liste.append([X,Y])
        
        C[j,:] = np.array([ X[-1],Y[-1] ])
        #plt.plot(X,Y,color="blue")

        j+=1
        
    #plt.plot(C[:,0],C[:,1],color=col,label='radius='+str(delta))
    
    return n,C,np.array(Liste)

def partial_Christoffel(alpha,beta):
    """Partial derivatives of the Christoffel symbols"""
    eps = 1.e-7
    Gam = Christoffel_symbs(alpha,beta)
    
    return (Christoffel_symbs(alpha +eps,beta)-Gam)/eps, (Christoffel_symbs(alpha,beta +eps)-Gam)/eps
    
def curvature_nontr(alpha,beta):
    """Approximation of the curvature for the Gamma family in the non-truncated case"""
    partial_Chr = partial_Christoffel(alpha,beta)
    partial_alpha = partial_Chr[0]
    partial_beta = partial_Chr[1]
    Gam = Christoffel_symbs(alpha,beta)
    I_mat = I(alpha,beta)
    
    R_121_1 = partial_alpha[0,0,0] - partial_beta[0,1,0] + np.sum(Gam[0,0,:]*Gam[1,:,0]) - np.sum(Gam[0,1,:]*Gam[0,:,0])
    R_121_2 = partial_beta[0,0,1] - partial_alpha[0,1,1] + np.sum(Gam[0,0,:]*Gam[1,:,1]) - np.sum(Gam[0,1,:]*Gam[0,:,1])
    
    return R_121_1*I_mat[0,1] + R_121_2*I_mat[1,1]
    
#######################################
########### Truncated case ############
#######################################

# À calculer à la main
def grad_Hess_logN(m,s,a,b):
    """Gradient and Hessian of the logarithm of the normalisation constant"""
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
    
   
def fun(y):
    return y*p(0,1,y)
    
    
def fun2(y):
    return y**2*np.exp(-2*y - np.exp(-y))

def fun1(y):
    return y*p(0,1,y)
    

def J(m,s,a,b):
    """Fisher information matrix for the Gamma family"""
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


# partial derivatives of the fisher metric

def partial_J(typ,m,s,a,b):
    """Partial derivatives of the Fisher information matrix"""
    eps = 1.e-7

    if typ == "m":
        return (J(m + eps,s,a,b) - J(m,s,a,b))/eps
    
    if typ == "s":
        return (J(m,s+eps,a,b) - J(m,s,a,b))/(eps)
    


# Christoffel symbols

def ind(i):
    if i==1:
        return "m"
    if i==2:
        return "s"
    
def J_inv(m,s,a,b):
    """Inverse of the Fisher information matrix"""
    J_inverse = np.linalg.inv(J(m,s,a,b))
    return J_inverse

def coef(i,s):
    if i==1:
        return 0
    else:
        return -2/s**3


def Gamma_Gum(i,j,k,m,s,a,b):
    """Christoffel symbols for the Gamma family"""
    J_inverse = J_inv(m,s,a,b)
    L = [0.5*J_inverse[k-1,l-1]*(partial_J(ind(j),m,s,a,b)[l-1,i-1] + partial_J(ind(i),m,s,a,b)[l-1,k-1] - partial_J(ind(l),m,s,a,b)[i-1,j-1]) for l in [1,2]]
    return np.sum(L)
    
    
# champ de vecteurs pour résoudre géodésiques

# Champ de vecteurs

def H_Gamma(m,s,dm,ds,a,b):
    """Vector field defining the geodesic equation in the Gamma family"""
    A = -Gamma_Gum(1,1,1,m,s,a,b)*dm**2 -2*Gamma_Gum(1,2,1,m,s,a,b)*dm*ds -Gamma_Gum(2,2,1,m,s,a,b)*ds**2
    B = -Gamma_Gum(1,1,2,m,s,a,b)*dm**2 -2*Gamma_Gum(1,2,2,m,s,a,b)*dm*ds -Gamma_Gum(2,2,2,m,s,a,b)*ds**2

    return np.array([dm,ds,A,B])


# geodesic solver

def geod_Gumb_tronquees(Tf,h,Y_0,a,b):
    """ODE solver for the geodesic equation"""
    N = np.int64(Tf/h)
    Y=np.zeros((N,4))
    Y[0,:] = Y_0
    
    for n in range(N-1):
        Y[n+1,:]= Y[n,:] + h*H_Gamma(Y[n,0],Y[n,1],Y[n,2],Y[n,3],a,b)
        
    return Y[:,0],Y[:,1]




def tracer_sphere_tr_avec_geod_Gumb(p,delta,nb_pts,h,a,b):
    """Helper function for computing geodesics from the same point"""
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

        print("j=",j)
        j+=1
        

    return n,C,np.array(Liste)


