import openturns as ot
import numpy as np
import scipy as scp
from scipy.optimize import minimize
from statsmodels.stats.weightstats import DescrStatsW


class Likelihood:
    """
    Class for likelihood functions between two probability measures on the real line. 
    Easier for storing the two distributions as well as the lower, upper and variance 
    bounds on the likelihood which are necessary for computing quantile confidence 
    intervals using Hoeffding's and Bennett's inequalities.

    Attributes
    ----------
    mu : ot.Distribution
        the target distribution
    
    mu_0 : ot.Distribution
        the sampling distribution
    
    lowerBound : float
        a lower bound on the likelihood ratio mu/mu_0
    
    upperBound : float
        an upper bound on the likelihood ratio mu/mu_0 (assuming it exists)
    
    varianceBound : float
        an upper bound on the variance of the likelihood ratio mu/mu_0 (assuming it exists)
    
    pushforward : bool
        if true, then mu and mu_0 are assumed to be obtained as pushforward measures by some (hidden) function G
    
    Parameters
    ----------
    mu : ot.Distribution
        the target distribution
    
    mu_0 : ot.Distribution
        the sampling distribution
    
    pushforward : bool
        if true, then mu and mu_0 are assumed to be obtained as pushforward measures by some (hidden) function G
    
    If ever mu and mu0 are given as pushforward measures by a complex function G of two measures
    P and P0, then it is not usually possible to compute the output likelihood dmu/dmu0. But, it is 
    still possible to estimate quantiles of mu using a sample from mu0 (i.e. using computer evaluations
    of G on P0) as well as build CIs for it. This then requires to make assumptions on the input likelihood
    dP/dP0. All of this can be handled in the same class Likelihood by using a boolean variable that we 
    call pushforward. If pushforward is False, then we have the usual mu and mu0, this will be
    its default value. If pushforward is True, now the assumptions are verified on the input 
    likelihood. 
    """

    # a static method for the rate function of Bennett's bound
    @staticmethod
    def h(u):
        if u == -1:
            return 0
        
        else:
            return (1+u)*np.log(1 + u) - u

    def __init__(self, mu, mu0, lowerBound = None , upperBound = None, varianceBound = None, pushforward = False):

        self.mu = mu
        self.mu0 = mu0
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.varianceBound = varianceBound
        self.pushforward = pushforward

    def getTargetDistribution(self):
        return self.mu
    
    def getInstrumentalDistribution(self):
        return self.mu0
    
    def sampleInstrumentalDistribution(self, n):
        Y_sample = self.mu0.getSample(n)
        Y_sample = np.array(Y_sample).transpose()[0]
        return Y_sample
    
    def getBounds(self):
        return self.lowerBound, self.upperBound

    def getVarianceBound(self):
        return self.varianceBound
    
    def likelihoodRatioFunction(self, y):
        mu = self.mu
        mu0 = self.mu0
        return mu.computePDF(y)/mu0.computePDF(y)
        
    def empiricalWeightedCDF(self, Y_sample, t, X_sample = []):
        """ 
        Computes the weighted empirical cdf. This function is taken from the online supplementary code 
        of the paper [Gauchy et al., 2022] in Technometrics.

        Y_sample (Sample class from OpenTURNS): value of the model G on the sample
        t (float): evaluation point for the empirical cdf
        
        output (float): the value of the empirical cdf built using sample and evaluated on t
        """

        boolean = self.pushforward

        if boolean == True:
            likelihoodValues = np.array([self.likelihoodRatioFunction(x) for x in X_sample])

        else:
            likelihoodValues = np.array([self.likelihoodRatioFunction(y) for y in Y_sample])

        indicator = Y_sample <= t

        return np.sum(likelihoodValues*indicator)/np.sum(likelihoodValues) 

    def empiricalWeightedCDFParallelized(self, Y_sample, t, X_sample = []):
        """ 
        Computes the weighted empirical cdf. This function is taken from the online supplementary code 
        of the paper [Gauchy et al., 2022] in Technometrics.

        Y_sample (array): value of the model G on the sample
        t (array): evaluation point(s) for the empirical cdf
        
        output (np.ndarray): the value(s) of the empirical cdf built using sample and evaluated on t
        """
        n = len(Y_sample)
        boolean = self.pushforward
        
        if boolean == True:
            likelihoodValues = np.array([self.likelihoodRatioFunction(x) for x in X_sample])  # Evaluate the likelihood ratio on the sample points
        
        else:
            likelihoodValues = np.array([self.likelihoodRatioFunction(y) for y in Y_sample])  # Evaluate the likelihood ratio on the sample points
    

        G = np.array([Y_sample, ] * len(t))  # duplicate the array Y_sample 
        T = np.array([t, ] * len(n))  # duplicate the points on which the empirical cdf is evaluated

        Bool = G <= T.transpose()       # how many times does Y_sample surpass t ?

        M = np.array([likelihoodValues, ] * len(t))     # duplicate the list containing the likelihood ratios
        B = M * Bool                     # compute the product under the summation
        
        return np.sum(B, axis=1) / np.sum(likelihoodValues)  


    def _quantile(self, alpha, Y_sample, X_sample = [], side="left"):

        # Aggregate over ties
        # df = pd.DataFrame(index=np.arange(len(self.weights)))
        # df["weights"] = self.weights
        # df["vec"] = vec
        # dfg = df.groupby("vec").agg("sum")
        # weights = dfg.values[:, 0]

        # values = np.asarray(dfg.index)

        # print("in likelihood module", X_sample)
        # we first compute the weights by distinguishing the two cases
        if self.pushforward == True:
            unnormalizedWeights = np.array([self.likelihoodRatioFunction(x) for x in X_sample])
        
        else:
            unnormalizedWeights = np.array([self.likelihoodRatioFunction(y) for y in Y_sample])

        weights = unnormalizedWeights/np.sum(unnormalizedWeights)

        cweights = np.cumsum(weights)
        totwt = cweights[-1]
        targets = alpha * totwt
        ii = np.searchsorted(cweights, targets)

        if side == "left":
            return np.sort(Y_sample)[ii]
        if side == "right":
            return np.sort(Y_sample)[ii+1]

        # Exact hits
        # jj = np.flatnonzero(np.abs(targets - cweights[ii]) < 1e-10)
        
        # jj = jj[ii[jj] < len(cweights) - 1]
        # rslt[jj] = (Y_sample[ii[jj]] + Y_sample[ii[jj] + 1]) / 2

        # return rslt


# if __name__ == "__main__":
#     l = Likelihood(ot.Normal(1), ot.Normal(1), 0.5, 3, 5, False)
#     # Y = np.arange(50)
#     Y = np.array(ot.Gumbel(1, 1).getSample(50)).transpose()[0]

#     # print(Y)
#     #Y = np.array([float(i) for i in range(1, 11)])
#     print(np.sort(Y))
#     print(l._quantile(0.5, Y, side="left"))
#     print(l._quantile(0.5, Y, side="right"))

#     quit()

    # build the empirical quantile from the weighted cdf
    def empiricalWeightedQuantile(self, Y_sample, alpha, X_sample = [], side="left"):
        """
        Helper function for computing the quantile estimator taken from statsmodels.
        
        Y_sample (np.array): the sample on which the empirical weighted quantile will
        be computed
        X_sample (np.array): the input sample on which a numerical function was evaluated to 
        obtain Y_sample (if self.pushforward==True)
        alpha (array): the quantile orders
        
        output (float): empirical quantile of order alpha

        For more details about how the quantile method from statsmodels works, see the 
        webpage: https://www.statsmodels.org/stable/_modules/statsmodels/stats/weightstats.html#DescrStatsW.quantile
        """ 

        # print("in likelihood module", X_sample)
        # compute the weights by distinguishing the two cases
        if self.pushforward == True:
            unnormalizedWeights = np.array([self.likelihoodRatioFunction(x) for x in X_sample])
        else:
            unnormalizedWeights = np.array([self.likelihoodRatioFunction(y) for y in Y_sample])
        

        weights = unnormalizedWeights/np.sum(unnormalizedWeights)

        # cweights = np.cumsum(weights)
        # totwt = cweights[-1]
        # targets = alpha * totwt
        # ii = np.searchsorted(cweights, targets)

        # if side == "left":
        #     return np.sort(Y_sample)[ii]
        # if side == "right":
        #     return np.sort(Y_sample)[ii+1]
        
        # and then statsmodels
        dstat = DescrStatsW(Y_sample, weights=weights)

        if type(alpha) == float:
            return dstat.quantile(alpha, return_pandas = False).transpose()[0]

        return dstat.quantile(alpha, return_pandas = False).transpose()


    def computeLikelihoodBounds(self, x0 = None, truncated = False, truncationBounds = None, method="bruteforce", disc_num=5000):
        """
        x0: starting point for the minimization and maximization problem for finding the lower
        and upper bounds. By default, it is given by the 50%-quantile of the perturbed distribution.
        truncated: boolean for determining if the distributions are truncated or not.
        truncationBounds: interval truncation
        method: compute the lower and upper bounds either with scipy or bruteforce (works well in the truncated case)

        return: the likelihood's lower, upper and variance bounds
        """

        f0 = self.mu0
        f_th = self.mu

        # get the lower and upper truncation bounds and define the starting point x0
        # for the optimization solver
        e=1e-6
        if truncated:
            lbT, ubT = truncationBounds[0], truncationBounds[1]
            bounds = [(lbT+e, ubT-e)]


        else:
            lbT, ubT = -np.inf, np.inf
            bounds = [(lbT, ubT)]

        if x0 == None:
            x0 = np.array(f_th.computeQuantile(0.5))[0] # default value for x0          

        if truncated and method == "bruteforce":
            X = np.linspace(lbT+0.0000001, ubT-0.0000001, disc_num)
            L = [f_th.computePDF(x)/f0.computePDF(x) for x in X]
            lowerBound = min(L)
            upperBound = max(L)

        elif method == "scipy":
            lowerBound = minimize(lambda x: f_th.computePDF(x)/f0.computePDF(x), x0=x0,
                                bounds=bounds).fun
            
            upperBound = minimize(lambda x: -f_th.computePDF(x)/f0.computePDF(x), x0=x0,
                                bounds=bounds).fun*(-1)
            
        varianceBound = scp.integrate.quad(lambda x: f_th.computePDF(x)**2/f0.computePDF(x),
                                           lbT, ubT)[0]
        
        #print("lower",lowerBound)
        #print("upper",upperBound)
        #print("variance",varianceBound)

        return lowerBound, upperBound, varianceBound

    def setLikelihoodBounds(self, tupleBounds):
        lowerBound, upperBound, varianceBound = tupleBounds
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.varianceBound = varianceBound

    # define two functions which compute Hoeffding's and Bennett's bounds
    def HoeffdingProbabilityBound(self, n, alpha, epsilon, intervalType = "two-sided") -> (float):

        """
        Likelihood class method which allows to compute Hoeffding's bound given:
        n (integer): sample size;
        alpha (float btw 0 and 1): quantile order;
        epsilon (float): interval "size";
        intervalType (string): determines whether the required interval is two-sided, left or right bounded;

        returns a (conservative) non-asymptotic confidence bound, obtained from Hoeffding's inequality,
        on the probability in question.
        """

        lowerBound, upperBound = self.getBounds()

        # simplifying notations
        a = lowerBound
        b = upperBound
        eps = epsilon

        # computing the exponential bounds
        #print((b-a)**2*(1-alpha-eps)**2)
        #print((b-a)**2*(alpha-eps)**2)
    
        exp_minus = np.exp(-2*n*eps**2/b**2)  # for Z^-
        exp_plus = np.exp(-2*n*eps**2/b**2)    # for -Z^+
        
        # return the required probability bound

        if intervalType == "two-sided":
            return 1 - exp_minus - exp_plus

        if intervalType == "left bound":
            return 1 - exp_plus

        if intervalType == "right bound":
            return 1 - exp_minus

    def BennettProbabilityBound(self, n, alpha, epsilon, intervalType = "two-sided") -> (float):
        """
        Python function which allows to compute Bennett's bound given:
        n (integer): sample size;
        alpha (float btw 0 and 1): quantile order;
        epsilon (float): interval "size";
        intervalType (string): determines whether the required interval is two-sided, left or right bounded;

        returns a (conservative) non-asymptotic confidence bound, obtained from Bennett's inequality,
        on the probability in question.
        """

        lowerBound, upperBound = self.getBounds()
        varianceBound = self.getVarianceBound() 

        # simplifying notations
        a = lowerBound
        b = upperBound
        eps = epsilon
        nu = varianceBound

        # computing the necessary constants
        nu_p = min(nu,b*alpha)*(1-alpha+eps)**2 + min(nu,b*(1-alpha))*(-alpha+eps)**2   # (alpha**2+eps**2 - alpha*eps)
        nu_m = min(nu,b*alpha)*(1-alpha-eps)**2 + min(nu,b*(1-alpha))*(-alpha-eps)**2

        a_p = b*(alpha-eps)
        b_m = b*(1-alpha-eps)
        #print("epsilon =", epsilon)
        #print(b_m<= -nu_m/epsilon)

        # if a == 0, then a division by zero will occur in the exponential bound, 
        # we replace the bound with its limit as a -> 0

        if a_p < 1e-2:
            exp_plus = np.exp(-n*eps**2/(2*nu_p))
        else:
            exp_plus = np.exp(-n*nu_p/a_p**2 * Likelihood.h(a_p*eps/nu_p))       # Z^+
        
        exp_minus = np.exp(-n*nu_m/b_m**2 * Likelihood.h(b_m*eps/nu_m))      # -Z^-
        #print("exp_plus = ", exp_plus)
        # return the required probability bound

        if intervalType == "two-sided":
            return 1 - exp_minus - exp_plus

        if intervalType == "left bound":
            return 1 - exp_plus

        if intervalType == "right bound":
            return 1 - exp_minus

    def HoeffdingVsBennett(self, n, alpha, epsilon, intervalType = "two-sided") -> (tuple):
        H = self.HoeffdingProbabilityBound(n, alpha, epsilon, intervalType=intervalType)
        B = self.BennettProbabilityBound(n, alpha, epsilon, intervalType=intervalType)

        return (H>B, B>=H)
    
    # build a function to compute epsilon (the interval "size") from the confidence level beta,
    # the sample size n and the quantile order alpha
    def determineEpsilon(self, beta, alpha, n, intervalType = "two-sided", CiBoundType = "Hoeffding"):

        """
        Function for computing the parameter epsilon (interval "size") given:\\
        
        beta (float): the required (conservative) confidence level;\\
        alpha (float btw 0 and 1): the quantile order;\\
        n (integer): the available sample's size;\\
        CiBoundType

        CiBoundType is a variable that specifies the type of confidence interval the user wants to compute:
        the built-in types are Hoeffding and Bennett.
        """

        # define the string containing the name of the probability bound function 
        # from Hoeffding or Bennett
        whosProbabilityBound = f"{CiBoundType}ProbabilityBound"

        # get the method for the specified CI bound
        attrib = getattr(self, whosProbabilityBound)

        # define the function whose root we will determine
        def P(epsilon):
            return attrib(n, alpha, epsilon, intervalType) - beta
        
        # find the root of the equation P(epsilon) = beta
        epsilonBound = min(alpha, 1-alpha) - 0.000001

        #try:
        result = scp.optimize.root_scalar(P, bracket = [0, epsilonBound]) 
        #if result.converged:
        #    print(f"epsilon computation converged ? {result.converged} with {result.function_calls} evaluations")
        #else:
        #    print(f"epsilon computation converged ? {result.converged}")
        #print("P(eps)=",P(result.root))
        #except ValueError:
        #    print(f"Sample size too small for given value of alpha = {alpha}, beta = {beta} and target distribution !")
            
        #else:
        print("eps= ", result.root)
        return result.root

    # build a function which computes the confidence interval
    def computeCI(
            self, Y_sample, alpha, beta, X_sample = [], intervalType = "two-sided", CiBoundType = "Hoeffding"
    ):

        n = len(Y_sample)

        # score = self.HoeffdingVsBennett(self, n, alpha, epsilon, intervalType=intervalType)
        # if score == (1, 0):
        #     CiBoundType = "Hoeffding"
        # elif score == (0, 1):
        #     CiBoundType = "Bennett"

        try:
            epsilon = self.determineEpsilon(beta, alpha, n, intervalType, CiBoundType)
        except ValueError:
            epsilon = 1
            print("eps= ", 1)
        # if epsilon == None:
        #     return None    #print(epsilon)
        #except:
        #    print(f"Sample size is not large enough for given value of alpha {alpha}, beta {beta} and interval type: {intervalType} !")
        #    return None
        
        # first, we deal with misspecified interval types
        if intervalType not in ["left bound", "right bound", "two-sided"]:
            print(f"The specified interval type {intervalType} cannot be handled")
            return None   # better way to break the function ?
        
        if intervalType == "left bound":
            if epsilon >= alpha:
                leftIntervalBound = np.nan
            else:
                leftIntervalBound = self.empiricalWeightedQuantile(
                    Y_sample, alpha-epsilon, X_sample = X_sample
                )
            rightIntervalBound = np.nan
        
        if intervalType == "right bound":
            if epsilon >= 1-alpha:
                rightIntervalBound = np.nan
            else:
                rightIntervalBound = self.empiricalWeightedQuantile(
                    Y_sample, alpha+epsilon, X_sample = X_sample, side='right'
                ) #/!\ right sided quantile !
            leftIntervalBound = np.nan

        if intervalType == "two-sided": 
            if epsilon >= alpha:
                leftIntervalBound = np.nan
            elif epsilon >= 1-alpha:
                rightIntervalBound = np.nan
            else:
                leftIntervalBound = self.empiricalWeightedQuantile(
                    Y_sample, alpha-epsilon, X_sample = X_sample
                )
                rightIntervalBound = self.empiricalWeightedQuantile(
                    Y_sample, alpha+epsilon, X_sample = X_sample, side='right'
                ) #/!\ right sided quantile !

        # check the probability bound for the computed epsilon
        whosProbabilityBound = f"{CiBoundType}ProbabilityBound"
        attrib = getattr(self, whosProbabilityBound)
        probabilityBound = attrib(n, alpha, epsilon, intervalType)

        return [leftIntervalBound, rightIntervalBound, probabilityBound, self.lowerBound, self.upperBound,
                self.varianceBound]
        
    def determineBeta(self, alpha, n, epsilon, intervalType = "two-sided", CiBoundType = "Hoeffding") -> (float):
        
        whosProbabilityBound = f"{CiBoundType}ProbabilityBound"

        # get the method for the specified CI bound
        attrib = getattr(self, whosProbabilityBound)

        return  attrib(n, alpha, epsilon, intervalType)

    def determineN(self, alpha, beta, epsilon, intervalType = "two-sided", CiBoundType = "Hoeffding"):

        whosProbabilityBound = f"{CiBoundType}ProbabilityBound"

        # get the method for the specified CI bound
        attrib = getattr(self, whosProbabilityBound)

        # define the function whose root we will find
        def P(n):
            return attrib(n, alpha, epsilon, intervalType) - beta
        
        # find the root of the equation P(n) = beta
        try:
            result = scp.optimize.root_scalar(P, bracket = [0, 10**8]) # better than 0.5 for the upper bound ?
            print(result)

        except ValueError:
            print(f"The specified parameters give a sample size larger than 10**8 !")

        else:
            return int(result.root) + 1


###===========================================###
###======== Previous versions ================###
###===========================================###

    # def empiricalWeightedQuantile(self, Y_sample, alpha):
    #     """
    #     Helper function for computing the quantile estimator. This method is faster than the previous one because
    #     it does not need to compute the value of the empirical cdf for all the sample points.
        
    #     Sample_input (np.array): sample from the truncated normal variable with parameter theta_0
    #     H_val (np.array): value of the model G on the sample
    #     theta_0 (np.array): initial parameter
    #     theta (np.array): target parameter for the importance sampling procedure 
    #     alpha (array): quantile order
    #     a,b (float): truncation bounds
        
    #     output (float): empirical quantile of order alpha
    #     """ 

    #     ## can be done using statmodels

    #     # likelihoodValues = np.array([self.likelihoodRatioFunction(y) for y in Y_sample])
    #     # normalizedLikelihoods = likelihoodValues / np.sum(likelihoodValues)

    #     Y_sample_ord = np.sort(Y_sample)
    #     i = int(alpha*len(Y_sample))
        
    #     # if the cdf is smaller than alpha on the ith element of the ordered sample H_val
    #     # we increment i
        
    #     if self.empiricalWeightedCDF(Y_sample,[Y_sample_ord[i]]) <  alpha:
    #         while self.empiricalWeightedCDF(Y_sample, [Y_sample_ord[i]]) < alpha:
    #             i = i+1
    #         return Y_sample_ord[i-1] # the -1 is because of python's indexing convention

    #     # if the cdf is larger than alpha on the ith element of the ordered sample H_val
    #     # we decrement i
    #     else:
    #         while self.empiricalWeightedCDF(Y_sample, [Y_sample_ord[i]])>= alpha:
    #             i = i-1
    #         return Y_sample_ord[i] #again we use i instead of i-1 because of python's indexing convention
