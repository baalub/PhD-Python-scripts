import openturns as ot
import numpy as np
import previous_modules.truncated_Gaussians as trGauss

import matplotlib.pyplot as plt
from truncatedDistribution import TruncatedDistribution

from normal import Normal

class LogNormal(ot.LogNormal):
    """
    In this class, we inherited the LogNormal class from OpenTURNS. The goal is to add an additional
    method to this class which allows to compute the Fisher Information of the lognormal family for 
    both the truncated and non-truncated case. This matrix is computed in the standard (mu, sigma) 
    parametrization.   
    """

    def _fisherInformationTruncatedCase(self, interval):
        m = self.getParameter()[0]
        s = self.getParameter()[1]
        a = interval.getLowerBound()[0]
        b = interval.getUpperBound()[0]

        # J is the FIM imported from the previously built truncated_Gaussian module

        return trGauss.J(m, s, np.log(a), np.log(b))

    

    def fisherInformation(self, *args):

        """FIM for the lognormal family in both the truncated and non truncated case

        m (float): parameter
        s (float): parameter
        a (float): parameter
        b (float): parameter

        output (np.ndarray): array of size 2x2 of the Fisher information matrix at point (m,s)""" 
        
        if args:
            return self._fisherInformationTruncatedCase(*args)
        
        s = self.getParameter()[1]
        i = np.zeros((2,2))
        
        i[0,0] = 1/(s**2)
        i[0,1] = 0
        i[1,0] = 0 
        i[1,1] = 2/s**2
        
        return i
    

    def _exponentialMapTruncatedCase(self, v, *args):
        """
        Exponential map on the truncated normal family
        """
        h = 0.01

        interval = args[0]
        lower = interval.getLowerBound()[0]
        upper = interval.getUpperBound()[0]
        a = np.log(lower)
        b = np.log(upper)

        m = self.getParameter()[0]
        s = self.getParameter()[1]

        X_0 = np.array([m, s, v[0], v[1]])
        m_t, s_t = trGauss.geod_tronquees(1, h, X_0, a, b)

        k = LogNormal(m_t[-1], s_t[-1])

        return TruncatedDistribution(k, interval)


    def exponentialMap(self, v, *args):
        """
        exponential map on the lognormal family. 
        the geodesics are approximated using ode.int from scipy.
        Same as the normal family
        """
        
        if args:
            return self._exponentialMapTruncatedCase(v, *args)
        
        h = 0.001

        m = self.getParameter()[0]
        s = self.getParameter()[1]

        X_0 = np.array([m, s, v[0], v[1]])

        m_t, s_t = trGauss.geod_non_tronquees(1, h, X_0)

        k = LogNormal(m_t[-1], s_t[-1])

        return k


    def _sampleFisherRaoSphereTrCase(self, delta, nbPts, interval):
        m = self.getParameter()[0]
        s = self.getParameter()[1]
        lower = np.log(interval.getLowerBound()[0])
        upper = np.log(interval.getUpperBound()[0])

        # define the corresponding normal distribution with the same paramters but with the log of interval bounds
        equiv_trnormal = TruncatedDistribution(Normal(m, s), ot.Interval(lower, upper))
        spherePointsList = equiv_trnormal.sampleFisherRaoSphere(delta, nbPts)

        return spherePointsList 

        # J = self._fisherInformationTruncatedCase(interval)
        # L = np.linspace(0,2*np.pi,nbPts, endpoint=False)
  
        # spherePointsList = []
        # for t in L:
        #     v = np.array([np.cos(t),np.sin(t)])
        #     l_J = np.sqrt(np.dot(np.transpose(v),np.dot(J,v)))
        #     v_J = delta*(v/l_J)

        #     # the following operation should be parallelized
        #     spherePoint = self._exponentialMapTruncatedCase(v_J, interval)

        #     spherePointsList.append(spherePoint)
            
        # return spherePointsList

    
    def sampleFisherRaoSphere(self, delta, nbPts, *args):
        """
        taken from the tracer_sphere_fisher_avec_geod function from the truncated_Gaussian module
        """

        if args:
            return self._sampleFisherRaoSphereTrCase(delta, nbPts, *args)
        
        
        I = self.fisherInformation()
        L = np.linspace(0,2*np.pi,nbPts, endpoint=False)
  
        spherePointsList = []
        for t in L:
            v = np.array([np.cos(t),np.sin(t)])
            l_J = np.sqrt(np.dot(np.transpose(v),np.dot(I,v)))
            v_J = delta*(v/l_J)
            spherePoint = self.exponentialMap(v_J)
            spherePointsList.append(spherePoint)
            
        return spherePointsList
    

if __name__ == "__main__":
    # baseline truncated lognormal distribution
    ln = TruncatedDistribution(LogNormal(0,1), ot.Interval(0.1,3))

    # compute fisher sphere
    lst = ln.sampleFisherRaoSphere(delta=0.1, nbPts=30)
    lst_arr = np.array([*lst])

    # plots
    L = np.linspace(0, 4, 100)
    plt.plot(L, [ln.computePDF(x) for x in L])

    for par in lst:
        plt.plot(L , [TruncatedDistribution(LogNormal(par[0], par[1]), ot.Interval(0.1, 3)).computePDF(x) for x in L], color="red")
    plt.show()

    plt.scatter([0], [1])
    plt.plot(lst_arr[:,0], lst_arr[:,1])
    plt.show()
