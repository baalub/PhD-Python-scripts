import openturns as ot
import numpy as np
import previous_modules.truncated_Gaussians as trGauss
import os

from multiprocessing import Pool
from truncatedDistribution import TruncatedDistribution

def parallel(args):
    v, h, interval, m, s = args

    lower = interval.getLowerBound()[0]
    upper = interval.getUpperBound()[0]

    X_0 = np.array([m, s, v[0], v[1]])

    m_t, s_t = trGauss.geod_tronquees(1, h, X_0, lower, upper)

    return m_t[-1], s_t[-1]

        
class Normal(ot.Normal):
    """
    In this class, we inherit the Normal class from OpenTURNS. The goal is to add an additional
    method to this class which allows to compute the Fisher Information of the normal family for 
    both the truncated and non-truncated case. Other additional methods exist as in computing
    the exponential map (geodesic endpoints) and Fisher-Rao spheres. All computation is performed 
    in the standard (mu, sigma) parametrization. Note that this is only valid for univariate normal
    distributions.   
    """

    # TODO class for the multivariate normal distributions.

    def _fisherInformationTruncatedCase(self, interval):
         
        m = self.getParameter()[0]
        s = self.getParameter()[1]
        lower = interval.getLowerBound()[0]
        upper = interval.getUpperBound()[0]

        # J is the FIM imported from the previously built truncated_Gaussian module

        return trGauss.J(m, s, lower, upper)
    
    def fisherInformation(self, *args):

        """FIM for the Normal family in both the truncated and non truncated case

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

    def _exponentialMapTruncatedCase(self, v, *args, **kwargs):
        """
        Exponential map on the truncated normal family
        """
        h = kwargs['h']
        interval = args[0]
        m = self.getParameter()[0]
        s = self.getParameter()[1]

        args = [(v, h, interval, m, s)]

        with Pool(processes=os.cpu_count()) as pool:
            sphereParams = pool.map(
                parallel, args, chunksize=5
            )

        m, s = sphereParams[0]
        k = Normal(m, s)

        return TruncatedDistribution(k, interval)


        # k = Normal(m_t[-1], s_t[-1])

        # return TruncatedDistribution(k, interval)

    def exponentialMap(self, v, *args, **kwargs):
        """
        exponential map on the normal family. 
        the geodesics are approximated using ode.int from scipy.
        """

        if not kwargs:
            h = 0.005
        else:
            h = kwargs['h']
        
        if args:
            return self._exponentialMapTruncatedCase(v, *args, h=h)
        
        m = self.getParameter()[0]
        s = self.getParameter()[1]

        X_0 = np.array([m, s, v[0], v[1]])

        m_t, s_t = trGauss.geod_non_tronquees(1, h, X_0)

        k = Normal(m_t[-1], s_t[-1])

        return k
        
    # def concentric_spheres(Liste_geod,n,nb_spheres):
    #     """Helper function for building the concentric spheres
        
    #     Liste_geod (list): Contains the list of geodesics starting from the same point
    #     n (int): number of steps in the geodesic approximation (it corresponds to Tf/h where h is 
    #     the step size and Tf is the final time for the geodesic)
    #     nb_spheres (int): number of spheres
        
    #     output: array where each element is a collection of discretized points on a concentric sphere"""
        
    #     liste_de_spheres_pt = []
    #     for i in range(1,nb_spheres+1):
    #         k=int(n*i/nb_spheres)-1
    #         XXk = [X[k] for X in Liste_geod[:,0]]
    #         YYk = [Y[k] for Y in Liste_geod[:,1]]
    #         liste_de_spheres_pt.append(np.array([XXk,YYk]))
            
    #     return np.array(liste_de_spheres_pt)

    def _sequentialExponentialMapTrCase(self, v, discretizeNum, *args):
        h = 0.01
        
        m = self.getParameter()[0]
        s = self.getParameter()[1]

        interval = args[0]
        lower = interval.getLowerBound()[0]
        upper = interval.getUpperBound()[0]

        X_0 = np.array([m, s, v[0], v[1]])
        m_t, s_t = trGauss.geod_tronquees(1, h, X_0, lower, upper)

        n = np.int64(1/h)
        meanCoord = [m_t[int(n*i/discretizeNum)-1] for i in range(1, discretizeNum+1)] 
        stdCoord = [s_t[int(n*i/discretizeNum)-1] for i in range(1, discretizeNum+1)]

        discretizedGeodesic = [TruncatedDistribution(Normal(meanCoord[i], stdCoord[i]), interval) for i in range(len(stdCoord))]

        return discretizedGeodesic

    def _sequentialExponentialMap(self, v, discretizeNum, *args):
        h = 0.01
        m = self.getParameter()[0]
        s = self.getParameter()[1]

        if args:
            return self._sequentialExponentialMapTrCase(v, discretizeNum, *args)
        
        # prepare the initial condition and compute the geodesic
        X_0 = np.array([m, s, v[0], v[1]])
        m_t, s_t = trGauss.geod_non_tronquees(1, h, X_0)

        # uniformly extracting points on the geodesic
        n = np.int64(1/h) # total number of discretization points for the geodesic
        meanCoord = [m_t[int(n*i/discretizeNum)-1] for i in range(1, discretizeNum+1)] 
        stdCoord = [s_t[int(n*i/discretizeNum)-1] for i in range(1, discretizeNum+1)]

        # Put the distributions (discretized geodesic) in a list
        discretizedGeodesic = [Normal(meanCoord[i], stdCoord[i]) for i in range(len(stdCoord))]
        discretizedGeodesic = [self] + discretizedGeodesic
        return discretizedGeodesic

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

    def _sampleFisherRaoSphereTrCase(self, delta, nbPts, interval, **kwargs):
        
        h = 0.001

        m = self.getParameter()[0]
        s = self.getParameter()[1]

        Vects = self._tangentVectors(interval, delta, nbPts)
    
        # TODO parallelize this loop using multiprocessing Pool
        args = [(v, h, interval, m, s) for v in Vects]

        with Pool(processes=os.cpu_count()) as pool:
            sphereParams = pool.map(
                parallel, args, chunksize=1
            )        
        
        if True:  #kwargs['bool_param']==False:
            spherePoints = []
            for m,s in sphereParams:
                k = Normal(m, s)
                tr_k = TruncatedDistribution(k, interval)
                spherePoints.append(tr_k)
                # the following operation should be parallelized
                #spherePoint = self._exponentialMapTruncatedCase(v_J, interval)
                #spherePointsList.append(spherePoint)
                
            return spherePoints
        else:
            return sphereParams
        

    def sampleFisherRaoSphere(self, delta, nbPts, *args, **kwargs):
        """
        taken from the tracer_sphere_fisher_avec_geod function from the truncated_Gaussian module
        """

        h = 0.001

        if args:
            return self._sampleFisherRaoSphereTrCase(delta, nbPts, *args, **kwargs)
        
        I = self.fisherInformation()
        L = np.linspace(0,2*np.pi, nbPts, endpoint=False)
  
        spherePointsList = []
        for t in L:
            v = np.array([np.cos(t),np.sin(t)])
            l_J = np.sqrt(np.dot(np.transpose(v),np.dot(I,v)))
            v_J = delta*(v/l_J)
            spherePoint = self.exponentialMap(v_J)
            spherePointsList.append(spherePoint)
            
        return spherePointsList


    def _fisher_rao_distance(self, f) -> float: 
        
        # TODO test the correctness of the computation
        
        p = self.getParameter()
        q = f.getParameter()

        p1, p2 = p[0]/np.sqrt(2), p[1]
        q1, q2 = q[0]/np.sqrt(2), q[1]
        
        N = np.sqrt( (p1- q1)**2 + (p2+q2)**2 ) + np.sqrt((p1- q1)**2 + (p2-q2)**2 ) 
        D = np.sqrt( (p1- q1)**2 + (p2+q2)**2 ) - np.sqrt((p1- q1)**2 + (p2-q2)**2 )
        
        return np.sqrt(2)*np.log(N/D)

    @classmethod
    def fisherRaoDistance(cls, f1, f2):
        """
        we want a class method for the Fisher-Rao distance since mathematically, the it is
        intrinsically depends on the parametric family.
        """

        # TODO finish error exceptions
        if type(f1) != cls or type(f2) != cls:
            raise TypeError(f"The given disttibutions are not {cls.__name__} distributions")
        
        return f1._fisher_rao_distance(f2)
        





if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()


    f1 = TruncatedDistribution(Normal(0, 1), ot.Interval(-2, 2))
        #f2 = TruncatedDistribution(Normal(3, 4))


    sphere = f1.sampleFisherRaoSphere(delta=0.3, nbPts=100, h=0.0001)
    for f in sphere:
        print(f.getParameter())

    current_time = timeit.default_timer()

    print(f"It took {current_time-start} seconds")
    # TODO how to adapt this class method to truncated Distributions ?

    # for i in range(3):
    #     def func(x):
    #         return x
        
    #     print(hex(id(func)))

    
