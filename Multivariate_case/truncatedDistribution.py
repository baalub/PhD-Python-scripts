import openturns as ot
import numpy as np
import scipy

class TruncatedDistribution(ot.TruncatedDistribution):
    """
    This custom class directly inherits the TruncatedDistribution class from OpenTURNS.
    """

    listOfPredefinedDists = ["Normal", "LogNormal", "Gumbel", "Gamma"]

    def __init__(self, base_dist, interval):
        # OpenTURNS TruncatedDistribution
        super().__init__(base_dist, interval)
        
        # Store the custom base distribution so we can access it from inside the class
        self._custom_base_distribution = base_dist


    def _integrand(self, u):
        # mettre *u
        # v = np.array(u)

        # Perform the quantile transform
        x = self.computeQuantile(u)

        # Determine the number of parameters
        D = len(self.getParameter()) 
        
        # Start building the matrix containing the integrand functions of the Fisher 
        # information
        integrands = np.zeros((D,D))

        for i in range(D):
            for j in range(i, D):
                integrands[i,j] = self.computeLogPDFGradient(x)[i]*self.computeLogPDFGradient(x)[j]

        return integrands
    

    def _fisherInformationGeneralCase(self):
        D = len(self.getParameter())
        I = np.zeros((D,D))

        for i in range(D):
            for j in range(i,D):
                I[i,j] = scipy.integrate.quad(lambda u: self._integrand(u)[i,j],0.0000001,0.9999999)[0]         

        J = I + I.transpose() - np.diag(I.diagonal())
        return J
    
    def fisherInformation(self):
        interval = self.getBounds()
        #lower = interval.getLowerBound()[0]
        #upper = interval.getUpperBound()[0]

        
        if str(self._custom_base_distribution.__class__.__name__) in TruncatedDistribution.listOfPredefinedDists:
            if not hasattr(self._custom_base_distribution, "fisherInformation"):
                raise AttributeError("The specified non-truncated distribution is an OpenTURNS distribution, not the custom base one")
            else:
                return self._custom_base_distribution.fisherInformation(interval)
                
        else:
            return self._fisherInformationGeneralCase()[:-2,:-2]


# if __name__ == "__main__":
#     from normal import Normal
#     from gumbel import Gumbel
#     f = TruncatedDistribution(ot.Beta(0.1, 0.1), ot.Interval(-1, 1))
#     print(f.fisherInformation())


    def exponentialMap(self, v, **kwargs):
        interval = self.getBounds()

        if not hasattr(self._custom_base_distribution, "exponentialMap"):
            raise AttributeError("The specified non-truncated distribution is an OpenTURNS distribution, not the custom base one")
        
        if kwargs:
            return self._custom_base_distribution.exponentialMap(v, interval, h=kwargs['h'])
        else:
            return self._custom_base_distribution.exponentialMap(v, interval)
    

    def _sequentialExponentialMap(self, v, discretizeNum):
        interval = self.getBounds()
        
        if not hasattr(self._custom_base_distribution, "_sequentialExponentialMap"):
            raise AttributeError("The specified non-truncated distribution is an OpenTURNS distribution, not the custom base one")
        
        return self._custom_base_distribution._sequentialExponentialMap(v, discretizeNum, interval)
    	
    
    def sampleFisherRaoSphere(self, delta, nbPts, **kwargs):
        interval = self.getBounds()

        if not hasattr(self._custom_base_distribution, "sampleFisherRaoSphere"):
            raise AttributeError("The specified non-truncated distribution is an OpenTURNS distribution, not the custom base one")

        if kwargs:
            return self._custom_base_distribution.sampleFisherRaoSphere(
                delta, nbPts, interval, bool_param=kwargs['bool_param']
                )
        else:
            return self._custom_base_distribution.sampleFisherRaoSphere(
                delta, nbPts, interval
                )
