import openturns as ot
import numpy as np


class Normal(ot.Normal):

    """
    In this class, we inherited the Normal class from OpenTURNS. The goal is to add an additional
    method to this class which allows to compute the Fisher Information of the normal family for 
    both the truncated and non-truncated case. This matrix is computed in the standard (mu, sigma)
    parametrization. Note that this is only valid for univariate normal
    distributions.   
    """

    def fisherInformationTruncatedCase(self, lower, upper):
        # write the expression of the Fisher information in the truncated case on [a,b]

        return "hello truncation"
        # J_B = np.zeros((2,2))

        # m = self.getParameter()[0]
        # s = self.getParameter()[1]

        # gradmu, gradsigma, Hess1, Hess2 = grad_hess_de_mu_sigma(m,s,a,b)

        # J_B[0,0] = gradmu[0] /(s**2)
        # J_B[0,1] = gradmu[1]/(s**2)
        # J_B[1,0] = J_B[0,1] 
        # J_B[1,1] = (gradsigma[1] + 2*(mu_B(m,s,a,b) - m) *gradmu[1] )/(s**3)
        
        # return J_B


    def fisherInformation(self, *args):

        """FIM for the Normal family in both the truncated and non truncated case

        m (float): parameter
        s (float): parameter
        a (float): parameter
        b (float): parameter

        output (np.ndarray): array of size 2x2 of the Fisher information matrix at point (m,s)""" 
        
        if args:
            return self.fisherInformationTruncatedCase(*args)
        
        s = self.getParameter()[1]
        i = np.zeros((2,2))
        
        i[0,0] = 1/(s**2)
        i[0,1] = 0
        i[1,0] = 0 
        i[1,1] = 2/s**2
        
        return i

