import openturns as ot
import numpy as np
import scipy as scp
from scipy.optimize import minimize



class Likelihood:
    """
    Class for likelihood functions between two probability measures on the real line. 
    Easier for storing the two distributions as well as the lower, upper and variance 
    bounds on the likelihood.
    """

    def __init__(self, mu, mu0, lowerBound, upperBound, varianceBound):
        self.mu = mu
        self.mu0 = mu0
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.varianceBound = varianceBound

    def getTargetDistribution(self):
        return self.mu
    
    def getInstrumentalDistribution(self):
        return self.mu0
    
    def getBounds(self):
        return self.lowerBound, self.upperBound

    def getVarianceBound(self):
        return self.varianceBound
    
    def likelihoodRatioFunction(self, y):
        return self.mu.computePDF(y)/self.mu0.computePDF(y)





# given a sample from mu0 of size n, build the empirical cdf of mu using the importance sampling
# weights given by the likelihood ratio mu/mu0

def likelihoodBounds(mu, mu0):

    """
    Python function for computing lower and upper bounds on the likelihood function as well as 
    approximating the variance of the likelihood variable using scipy.optimize. Since, most
    distributions that we will be using are simple, the likelihood ratio will be fairly easy
    to optimize. But if the bounds can be computed analytically then the analytical bounds is 
    preferred.
    """ 

    lowerBound = minimize(lambda y: likelihood(mu, mu0, y), 0).fun
    upperBound = minimize(lambda y: -likelihood(mu, mu0, y), 0).fun *(-1)

    variance = scp.integrate.quad(lambda y: likelihood(mu, mu0, y), -np.inf, np.inf)[0]

    return lowerBound, upperBound, variance

def empiricalWeightedCDF(mu, mu0, Y_sample, t):
    pass

# build the empirical quantile from the weighted cdf
def empiricalWeightedQuantile(mu, mu0, Y_sample, alpha):
    pass

# define a function which computes Hoeffding's bound

def HoeffdingProbabilityBound(mu, mu0, n, alpha, epsilon, intervalType = "two-sided") -> (float):

    """
    Python function which allows to compute Hoeffding's bound given:\\
    n (integer): sample size;\\
    alpha (float btw 0 and 1): quantile order;\\
    epsilon (float): interval "size";\\
    intervalType (string): determines whether the required interval is two-sided, left or right bounded;\\

    returns a (conservative) non-asymptotic confidence bound, obtained from Hoeffding's inequality,
    on the probability in question.
    """

    lowerBound, upperBound, variance = likelihoodBounds(mu, mu0)

    # simplifying notations

    a = lowerBound
    b = upperBound
    eps = epsilon

    # computing the exponential bounds

    exp_moins = np.exp(-2*n*eps**2/((b-a)**2*(1-alpha-eps)**2))  # for Z^-
    exp_plus = np.exp(-2*n*eps**2/((b-a)**2*(alpha-eps)**2))    # for -Z^+
    
    # return the required probability bound

    if intervalType == "two-sided":
        return 1 - exp_moins - exp_plus

    if intervalType == "left bound":
        return 1 - exp_plus

    if intervalType == "right bound":
        return 1 - exp_moins


# define the necessary rate function for Bennett bound

def h(u):
    return (1+u)*np.log(1 + u) - u

def BennettProbabilityBound(mu, mu0, n, alpha, epsilon, intervalType = "two-sided") -> (float):

    """
    Python function which allows to compute Bennett's bound given:\\
    n (integer): sample size;\\
    alpha (float btw 0 and 1): quantile order;\\
    epsilon (float): interval "size";\\
    intervalType (string): determines whether the required interval is two-sided, left or right bounded;\\

    returns a (conservative) non-asymptotic confidence bound, obtained from Bennett's inequality,
    on the probability in question.
    """

    lowerBound, upperBound, varianceBound = likelihoodBounds(mu, mu0)

    # simplifying notations

    a = lowerBound
    b = upperBound
    eps = epsilon
    nu = varianceBound

    # computing the necessary constants

    nu_p = min(nu,b*alpha)*(1-alpha+eps)+nu*(alpha**2+eps**2 - alpha*eps)
    nu_m = min(nu,b*alpha)*(1-alpha-eps)+nu*(alpha**2+eps**2 + alpha*eps)

    a_p = -a*(-alpha+eps)
    b_m = b*(1-alpha-eps)
    

    # if a == 0, then a division by zero will occur in the exponential bound, 
    # we replace the bound with its limit as a -> 0

    if a == 0:
        exp_plus = np.exp(-n*eps**2/(2*nu_p))

    exp_plus = np.exp(-n*nu_p/a_p**2 * h(a_p*eps/nu_p))       # Z^+
    exp_moins = np.exp(-n*nu_m/b_m**2 * h(b_m*eps/nu_m))      # -Z^-


    # return the required probability bound

    if intervalType == "two-sided":
        return 1 - exp_moins - exp_plus

    if intervalType == "left bound":
        return 1 - exp_plus

    if intervalType == "right bound":
        return 1 - exp_moins

# build a function to compute epsilon (the interval "size") from the confidence level beta,
# the sample size n and the quantile order alpha
def determineEpsilon(beta, alpha, n, intervalType = "two-sided", CiBoundType = "Hoeffding"):

    """
    Python function for computing the parameter epsilon given:\\
    
    beta (float): the required (conservative) confidence level;\\
    alpha (float btw 0 and 1): the quantile order;\\
    n (integer): the available sample's size;\\
    CiBoundType

    CiBoundType is a variable that specifies the type of confidence interval the user wants to compute:
    the built-in types are Hoeffding and Bennett.
    """

    if CiBoundType == "Hoeffding":
        pass

    if CiBoundType == "Bennett":
        pass

# build a function which computes the confidence interval
def computeCI(mu, mu0, Y_sample, alpha, beta, intervalType = "two-sided", CiBoundType = "Hoeffding"):

    n = len(Y_sample)

    lowerBoundL, upperBoundL, varianceL = likelihoodBounds(mu, mu0)
    epsilon = determineEpsilon(beta,alpha, n, intervalType, CiBoundType)

    leftIntervalBound = empiricalWeightedQuantile(mu, mu0, Y_sample, alpha-epsilon)
    quantileEstimator = empiricalWeightedQuantile(mu, mu0, Y_sample, alpha)
    rightIntervalBound = empiricalWeightedQuantile(mu, mu0, Y_sample, alpha+epsilon) ### this is not exactly the right interval bound

    if CiBoundType == "Hoeffding":
        b = HoeffdingProbabilityBound(mu, mu0, n, alpha, epsilon,  )
        return [leftIntervalBound, rightIntervalBound, quantileEstimator, ]

