import openturns as ot
import numpy as np
import previous_modules.truncated_Gumbel as trGumb
import os

from multiprocessing import Pool
from truncatedDistribution import TruncatedDistribution

def parallel(args):
    v, h, interval, m, s = args
    print("parallel")
    lower = interval.getLowerBound()[0]
    upper = interval.getUpperBound()[0]

    X_0 = np.array([m, s, v[0], v[1]])

    m_t, s_t = trGumb.geod_Gumb_tronquees(1, h, X_0, lower, upper)

    return m_t[-1], s_t[-1]


class Gumbel(ot.Gumbel):

    def _fisherInformationTruncatedCase(self, interval):

	# the order m and s is inverted to respect openturns convention for gumbel distributions 
        m = self.getParameter()[1]
        s = self.getParameter()[0]
        lower = interval.getLowerBound()[0]
        upper = interval.getUpperBound()[0]

        # J is the FIM imported from the previously built truncated_Gumbel module
	
        return trGumb.J(m, s, lower, upper)

    def fisherInformation(self, *args):

        """FIM for the Gumbel family in both the truncated and non truncated case

        m (float): parameter
        s (float): parameter
        a (float): parameter
        b (float): parameter

        output (np.ndarray): array of size 2x2 of the Fisher information matrix at point (m,s)""" 
        
        if args:
            return self._fisherInformationTruncatedCase(*args)

        m = self.getParameter()[0]        
        s = self.getParameter()[1]

        # computation based on scipy.integrate

        return trGumb.I(m, s)


    def _exponentialMapTruncatedCase(self, v, *args):
        """
        Exponential map on the truncated Gumbel family
        """
        h = 0.01
        interval = args[0]

        lower = interval.getLowerBound()[0]
        upper = interval.getUpperBound()[0]
        m = self.getParameter()[0]
        s = self.getParameter()[1]

        X_0 = np.array([m, s, v[0], v[1]])
        m_t, s_t = trGumb.geod_Gumb_tronquees(1, h, X_0, lower, upper)
        
        return np.array([m_t[-1], s_t[-1]])
    
        # k = Gumbel(m_t[-1], s_t[-1])
        # return TruncatedDistribution(k, interval)

    def exponentialMap(self, v, *args):
        """
        exponential map on the Gumbel family. 
        the geodesics are approximated using ode.int from scipy.
        """
        h=0.01
        print("args =", args)
        
        if args:
            return self._exponentialMapTruncatedCase(v, *args)
        
        m = self.getParameter()[0]
        s = self.getParameter()[1]

        X_0 = np.array([m, s, v[0], v[1]])
        print("in exp map", type(X_0))
        m_t, s_t = trGumb.geod_Gumb_non_tronquees(1, h, X_0)

        k = Gumbel(m_t[-1], s_t[-1])

        return k
    
    def _sequentialExponentialMapTrCase(self, v, discretizeNum, *args, **kwargs):
        h = 0.01
        
        m = self.getParameter()[1]
        s = self.getParameter()[0]

        interval = args[0]
        lower = interval.getLowerBound()[0]
        upper = interval.getUpperBound()[0]

        X_0 = np.array([m, s, v[1], v[0]])
        m_t, s_t = trGumb.geod_Gumb_tronquees(1, h, X_0, lower, upper)

        n = np.int64(1/h)
        meanCoord = [m_t[int(n*i/discretizeNum)-1] for i in range(1, discretizeNum+1)] 
        stdCoord = [s_t[int(n*i/discretizeNum)-1] for i in range(1, discretizeNum+1)]
        
        if kwargs['bool_param'] == False:
            return meanCoord, stdCoord
        
        else:
            discretizedGeodesic = [TruncatedDistribution(Gumbel(stdCoord[i], meanCoord[i]), interval) for i in range(len(stdCoord))]
            return discretizedGeodesic
        
    def _sequentialExponentialMap(self, v, discretizeNum, *args):
        h = 0.01
        m = self.getParameter()[1]
        s = self.getParameter()[0]

        if args:
            return self._sequentialExponentialMapTrCase(v, discretizeNum, *args)
        
        # prepare the initial condition and compute the geodesic
        X_0 = np.array([m, s, v[0], v[1]])
        m_t, s_t = trGumb.geod_Gumb_non_tronquees(1, h, X_0)

        # uniformly extracting points on the geodesic
        n = np.int64(1/h) # total number of discretization points for the geodesic
        meanCoord = [m_t[int(n*i/discretizeNum)-1] for i in range(1, discretizeNum+1)] 
        stdCoord = [s_t[int(n*i/discretizeNum)-1] for i in range(1, discretizeNum+1)]

        # Put the distributions (discretized geodesic) in a list
        discretizedGeodesic = [Gumbel(stdCoord[i], meanCoord[i]) for i in range(len(stdCoord))]

        return discretizedGeodesic

        
#     def _sampleFisherRaoSphereTrCase(self, delta, nbPts, interval):
        
#         h = 0.01
# #        print("interval=", type(interval))
#         J = self._fisherInformationTruncatedCase(interval)
#         L = np.linspace(0,2*np.pi,nbPts, endpoint=False)
  
#         spherePointsList = []
        
#         # TODO parallelize this loop using multiprocessing Pool
#         for t in L:
#             v = np.array([np.cos(t),np.sin(t)])
#             l_J = np.sqrt(np.dot(np.transpose(v),np.dot(J,v)))
#             v_J = delta*(v/l_J)

#             # the following operation should be parallelized
#             spherePoint = self._exponentialMapTruncatedCase(v_J, interval)

#             spherePointsList.append(spherePoint)
            
#         return spherePointsList

    def _sampleFisherRaoSphereTrCase(self, delta, nbPts, interval, **kwargs):
        
        h = 0.01

        m = self.getParameter()[1]
        s = self.getParameter()[0]

        Vects = self._tangentVectors(interval, delta, nbPts)
    
        # TODO parallelize this loop using multiprocessing Pool
        args = [(v, h, interval, m, s) for v in Vects]

        with Pool(processes=os.cpu_count()) as pool:
            sphereParams = pool.map(
                parallel, args, chunksize=1
            )        

        return sphereParams


    def _tangentVectors(self, interval, delta, nbPts):   

        J = self._fisherInformationTruncatedCase(interval)
        L = np.linspace(0, 2*np.pi, nbPts, endpoint=False)
        Vects = []

        for t in L:
            v = np.array([np.cos(t),np.sin(t)])
            l_J = np.sqrt(np.dot(np.transpose(v),np.dot(J,v)))
            v_J = delta*(v/l_J)
            Vects.append(v_J)
        
        return Vects
    

    def sampleFisherRaoSphere(self, delta, nbPts, *args):
        """
        taken from the tracer_sphere_fisher_avec_geod function from the truncated_Gaussian module
        """

        if args:
            return self._sampleFisherRaoSphereTrCase(delta, nbPts, *args)
        
        I = self.fisherInformation()
        L = np.linspace(0,2*np.pi,nbPts, endpoint=False)
  
        spherePointsList = []

        # TODO parallelize this loop using multiprocessing Pool
        for t in L:
            v = np.array([np.cos(t),np.sin(t)])
            l_J = np.sqrt(np.dot(np.transpose(v),np.dot(I,v)))
            v_J = delta*(v/l_J)
            spherePoint = self.exponentialMap(v_J)
            spherePointsList.append(spherePoint)
            
        return spherePointsList
