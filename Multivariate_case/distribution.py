import openturns as ot
import scipy
import numpy as np

class Distribution(ot.Distribution):

    """
    This class inherits the Distribution class from OpenTURNS and gives an additional method to 
    compute the Fisher Information for the family containing the distribution.

    If the distribution is elliptical, then we can apply a simplier formula for computing the Fisher
    information.
    """

    def _integrand(self, u):
        # mettre *u
        # v = np.array(u)

        # Perform the quantile transform

        x = self.computeQuantile(u)
        # x = self.computeSequentialConditionalQuantile(v)

        # Determine the number of parameters

        D = len(self.getParameter()) 
        
        # Start building the matrix containing the integrand functions of the Fisher 
        # information
            
        integrands = np.zeros((D,D))

        for i in range(D):
            for j in range(D):
                integrands[i,j] = self.computeLogPDFGradient(x)[i]*self.computeLogPDFGradient(x)[j]

        return integrands

    def fisherInformation(self):
        
        D = len(self.getParameter())
        I = np.zeros((D,D))

        if self.isElliptical():
            # Improve the computation of the Fisher information
            pass

        for i in range(D):
            for j in range(i,D):
                
                #def g(*u):
                #    return self.integrand(*u)[i,j]

                I[i,j] = scipy.integrate.quad(lambda u: self.integrand(u)[i,j],0.0000001,0.9999999)[0]         

                # scipy.integrate.quadvec
        J = I + I.transpose() - np.diag(I.diagonal())

        return J


    def exponentialMap(self, v, *args):
        """
        This method allows to compute the exponential map at the object (the distribution)
        in the v direction. Here v is an ndarray which corresponds vector's tangent space 
        coordinates in the statistical manifold of the family.
        """
    
        if args:
            return self.exponentialMapTruncatedCase(v, *args)
        
        h = 0.01

        m = self.getParameter()[0]
        s = self.getParameter()[1]

        X_0 = np.array([m, s, v[0], v[1]])
        # TODO
        m_f, s_f = 0, 0

        # get the name of the class of which self is an instance
        distType = self.__class__

        k = Distribution(distType(m_f, s_f))

        return k
        
    

def FisherRaoDistance(f1,f2):
    
    if type(f1)!=type(f2):
        print("the two distributions must be in the same family")
    else:
        pass

