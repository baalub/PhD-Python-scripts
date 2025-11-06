import openturns as ot
import numpy as np

from extendedtrnormal import ExtendedTrNormal

import previous_modules.xtrnormal as xtrnorm

class ExtendedTrLogNormal:

    def __init__(self, k, s, lowerBound, upperBound, description='X0'):
        """
        the lower and upper bounds of the extended log normal should be given in log-scale
        """

        self.k = k
        self.s = s
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.description = description

        #self.mu = ExtendedTrNormal._changeOfVariablesBack(k, s)[0]
        #self.sigma = ExtendedTrNormal._changeOfVariablesBack(k, s)[1]

        self._tr_interval = ot.Interval(self.lowerBound, self.upperBound)

    def __repr__(self):
        return f"ExtendedTrLogNormal(k={self.k}, s={self.s})\n"
    
    def getParameter(self, with_bounds=False):
        if with_bounds:
            return [self.k, self.s, self.lowerBound, self.upperBound]
        else:
            return [self.k, self.s]
    
    def setDescription(self, description: ot.Description):
        self.description = description

    def getDescription(self):
        return self.description
    
    def computePDF(self, x):

        k = self.k
        s = self.s
        a = self.lowerBound
        b = self.upperBound

        return xtrnorm.f_lognormal(x, k, s, a, b)

    def computeLogPDFGradient(self, x: ot.Sample):
        k = self.k
        s = self.s
        a = self.lowerBound
        b = self.upperBound

        f = ExtendedTrNormal(k, s, np.log(a), np.log(b))
        x_arr = np.array(x)
        log_x_arr = np.log(x_arr)
  
        return f.computeLogPDFGradient(ot.Sample(log_x_arr))
    
    def getSample(self, sampleSize):
        f = ExtendedTrNormal(self.k, self.s, np.log(self.lowerBound), np.log(self.upperBound))
        s_arr = np.array(f.getSample(sampleSize)).transpose()[0]
        s_lognormal_arr = np.exp(s_arr)
        s = ot.Sample(s_lognormal_arr.reshape(-1, 1))
        return s
    
if __name__ == "__main__":
    f = ExtendedTrLogNormal(0, -1, 0.1, 1)
    x = f.getSample(5)
    print('sample=', x)
    print('logpdfgrad=', f.computeLogPDFGradient(x))
    quit()
        # return ot.Sample(np.asarray([0 for _ in range(sampleSize)]).reshape(-1, 1))

if __name__ == "__main__":
    g = ExtendedTrNormal(0, 1, -1, 1)
    f = ExtendedTrLogNormal(0, 1, 0.1, 1)
    X = f.getSample(5)
    Y = g.getSample(5)
    print(X)
    print(Y)
