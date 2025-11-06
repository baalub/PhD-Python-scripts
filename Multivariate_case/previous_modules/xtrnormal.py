import openturns as ot
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from scipy.integrate import odeint


def n_dens(x, k, s, a=-1, b=1):

    if x < a:
        return 0
    if x > b:
        return 0

    return np.exp(-x**2*s/2 + k*x)   

def n_dens_lognormal(x, k, s, a=np.exp(-1), b=np.exp(1)):
    if x < a:
        return 0
    if x > b:
        return 0
    
    return np.exp(-np.log(x)**2*s/2 + k*x)/x


def normalizing_constant(k, s, a=-1, b=1):

    return integrate.quad(n_dens, a, b, args = (k, s, a, b))[0]
#    return integrate.quad(integrand, a, b)[0]

def normalizing_constant_lognormal(k, s, a=np.exp(-1), b=np.exp(1)):
    return integrate.quad(n_dens_lognormal, a, b, args = (k, s, a, b))[0]


def f(x, k, s, a, b): 
    """probability density function of extended normal"""

    return n_dens(x, k, s, a=a, b=b)/normalizing_constant(k, s, a=a, b=b)

def f_lognormal(x, k, s, a=np.exp(-1), b=np.exp(1)): 
    """probability density function of extended lognormal"""

    return n_dens_lognormal(x, k, s, a=a, b=b)/normalizing_constant_lognormal(k, s, a=a, b=b)

def func_gen(i, *args):
    k, s, a, b = args
    _norm = normalizing_constant(k, s, a, b)

    def func(x):
        return x**i*n_dens(x, k, s, a, b)/_norm
    
    return func

def mean(k, s, a=-1, b=1):
    func = func_gen(1, k, s, a, b)
    return integrate.quad(func, a, b)[0]

def second_moment(k, s, a=-1, b=1):
    func = func_gen(2, k, s, a, b)
    return integrate.quad(func, a, b)[0]

def skew(k, s, a=-1, b=1):
    func = func_gen(3, k, s, a, b)
    return integrate.quad(func, a, b)[0]

def kurtosis(k, s, a=-1, b=1):
    func = func_gen(4, k, s, a, b)
    return integrate.quad(func, a, b)[0]

def _statistic(x):
    return np.array([x, -x**2/2])
#    return np.array([-x**2/2, x])

def logpdf_gradient(x: ot.Sample, k, s, a=-1, b=1):
    
    if s>0:
        mu, sigma = k/s, 1/np.sqrt(s) 
        N = ot.Normal(mu, sigma)
        dist = ot.TruncatedDistribution(N, ot.Interval(a, b))
        loggrad = np.array(dist.computeLogPDFGradient(x))[:, 0:-2]
        return ot.Sample(loggrad)
    
    # compute the first and second moment of the (k,s)-distribution
    _mean_ = mean(k, s, a=a, b=b)
    _second_moment_ = second_moment(k, s, a=a, b=b)

    # duplicate the moments
    arr1 = np.array([_mean_, -0.5*_second_moment_])
    arr2 = np.repeat(arr1[:, None], len(x), axis=1)

    # compute the gradient (in the exponential family)
    stat_mat = _statistic(np.array(x).transpose()[0])
    log_pdf_grad = stat_mat - arr2
    # should the gradient be null if x is outside of the interval ?

    return ot.Sample(log_pdf_grad.transpose())

if __name__ == "__main__":
    x = np.linspace(-1, 1, 10)
    print(logpdf_gradient(x, 0, 0))

def information(k, s, a, b):

    _mean = mean(k, s, a=a, b=b)
    _second_moment = second_moment(k, s, a=a, b=b)
    _skew = skew(k, s, a=a, b=b)
    _kurtosis = kurtosis(k, s, a=a, b=b)

    I = np.zeros((2, 2))

    I[0, 0] = _second_moment - _mean**2
    I[1, 0] = -(_skew - _mean*_second_moment)/2
    I[0, 1] = I[1, 0]
    I[1, 1] = (_kurtosis - _second_moment**2)/4

    return I

def partial_i(k, s, a, b):
    eps = 1.e-7

    partial_k = (information(k+eps, s, a=a, b=b) - information(k-eps, s, a=a, b=b))/(2*eps)
    partial_s = (information(k, s+eps, a=a, b=b) - information(k, s-eps, a=a, b=b))/(2*eps)

    return np.array([partial_k, partial_s])

    

def christoffel_symbols(k, s, a, b):
    I = information(k, s, a=a, b=b)

    inverse = np.linalg.inv(I)
    partials=partial_i(k, s, a=a, b=b)

    Gam = np.zeros((2,2,2))
    
    for k in range(2):
        for j in range(2):
            for i in range(2):
                Gam[i,j,k] = np.sum(
            [0.5*inverse[k,l] *( partials[j][l,i] + partials[i][l,j] - partials[l][i,j] ) for l in range(2)]
            )
    
    return Gam


def H(t, V, a, b):

    k, s, v1, v2 = V[0], V[1], V[2], V[3]
    
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

    Gam_LC = christoffel_symbols(k, s, a, b)
    A1 = -Gam_LC[0,0,0]*v1**2 -2*Gam_LC[0,1,0]*v1*v2 -Gam_LC[1,1,0]*v2**2
    B1 = -Gam_LC[0,0,1]*v1**2 -2*Gam_LC[0,1,1]*v1*v2 -Gam_LC[1,1,1]*v2**2

    return np.array([v1, v2, A1, B1])


def geodesic(Tf, h, Y_0, a=-1, b=1):
    """
    ODE solver (Euler method) for the geodesic equation in the truncated case
    
    Tf (float): Corresponds to the endpoint of the interval on which the equation is solved i.e. [0,Tf]
    h (float): step size for Euler method
    X_0 (array): initial conditions
    a,b (floats): truncation bounds

    output (tuple): a tuple of two arrays corresponding to the x and y coordinates of the approximated geodesic
    """

    N = np.int64(Tf/h)
    #Y=np.zeros((N,4))
    # Y[0,:] = Y_0

    Y = integrate.solve_ivp(
         H, (0, 1), Y_0, t_eval=np.linspace(0, Tf, N, endpoint=True),
         method='BDF', args=(a,b) , rtol=1e-6, atol=1e-8
         )

    # Y_0, , args=(a,b))

    # for n in range(N-1):
    #     Y[n+1,:] = Y[n,:] + h*H(Y[n,0], Y[n,1], Y[n,2], Y[n,3], a, b)
    #print(np.shape(Y.y))
    return Y.y[0,:], Y.y[1,:]



if __name__ == "__main__":
    quit()    # print(f(0.1, 0, 1))

    k, s = -1, 2

    v1 = 2
    v2 = 0

    Y_0 = np.array([k, s, v1, v2])

    k_t, s_t = geodesic(Tf=1, h=0.01, Y_0=Y_0)

    print(k_t)
    print(s_t)

    # print(mean(k, s))
    # print(second_moment(k, s))
    # print(skew(k, s))
    # print(kurtosis(k, s))

    # print(I)
    # print("det = ", np.linalg.det(I))

    c = [(x, 0, 0) for x in np.linspace(0.2, 1, len(k_t))]

    # print(christoffel_symbols(0, 0))

    plt.figure(figsize=(10, 6))
    X = np.linspace(-1.2, 1.2, 300)

    for l in range(len(k_t)):
        if l%7==0:
            plt.plot(X, [f(x, k_t[l], s_t[l]) for x in X], lw = 2, alpha=0.8, color = c[l], label=f"(k,s)=({k_t[l]:.2f}, {s_t[l]:.2f})")
            plt.legend(ncols = 2)
            plt.savefig("/home/bketema/Documents/CATHARE/prez DT figures/extended_normal_geod_1.pdf", dpi=600, bbox_inches='tight')

    plt.show()


    plt.figure(figsize=(10, 6))

    
