import openturns as ot
import numpy as np
import scipy
import scipy.linalg

from normal import Normal
from extendedtrnormal import ExtendedTrNormal
from extendedtrlognormal import ExtendedTrLogNormal

class JointDistribution(ot.JointDistribution):

    def __init__(self, marginals, copula):

        _extrnormals = {}
        _extrlognormals = {}
        _rest = {}

        # the marginals that are not OpenTURNS distributions are stored separetely
        try:
            super().__init__(marginals, copula)

        except TypeError:
            
            for i, f in enumerate(marginals):

                if f.__class__.__name__ == "ExtendedTrNormal":
                    _extrnormals[i] = f

                elif f.__class__.__name__ == "ExtendedTrLogNormal":
                    _extrlognormals[i] = f

                else:
                    _rest[i] = f

        self.marginals = marginals
        self.copula = copula

        self._extrnormals = _extrnormals
        self._extrlognormals = _extrlognormals
        self._rest = _rest

        _description = ot.Description([f.getDescription()[0] for f in marginals])
        self._description = _description

        self._dimension = len(marginals)

    # def getDescription(self):
    #     return self._description
    
    @staticmethod
    def merge_samples(samples):
        return ot.Sample(np.hstack(samples))
    
    def getSingleMarginal(self, index):
        _extrnormals = self._extrnormals 
        _extrlognormals = self._extrlognormals
        _rest = self._rest

        if index in _extrlognormals.keys():
            return _extrlognormals[index]
        if index in _extrnormals.keys():
            return _extrnormals[index]
        if index in _rest.keys():
            return _rest[index]

    def getDimension(self):
        return self._dimension
    
    def getParameter(self):
        parameters = []
        for f in self.marginals:
            parameters.append(f.getParameter())
        
        parameters.append(self.copula.getParameter())
        return parameters
    
    # def getSample(self, sampleSize):
    #     marginals = self.marginals
    #     d = len(marginals)

    #     # if self.copula == ot.IndependentCopula(d):
    #     samples = [f.getSample(sampleSize) for f in marginals]
    #     s = JointDistribution.merge_samples(samples)

    #     desc = ot.Description(self.getDescription())
    #     s.setDescription(desc)
    #     return s
        
    #     # else:
    #     #    print("still haven't coded the case of dependence btw inputs")

    def getSample(self, sampleSize):
        marginals = self.marginals
        d = len(marginals)

        samples = [f.getSample(sampleSize) for f in marginals]
        s = JointDistribution.merge_samples(samples)

        s.setDescription(self.getDescription())
        return s
        
    # def computePDF(self, x: ot.Sample):
    #     # we assume that the copula is independent
    #     marginals = self.marginals
    #     pdf_current_val = np.ones((x.getSize(), 1))

    #     for i, f in enumerate(marginals):
    #         pdf_current_val* np.array(f.computePDF(x.getMarginal(i)))

    #     return pdf_current_val

    # def computePDF(self, *args):
    #     arr = np.array(super().computeLogPDFGradient(*args))[:, 0:-2]
    #     return ot.Sample(arr) 
    

    def computeLogPDFGradient(self, x: ot.Sample, stack=True):
        grad_blocks = []

        for f in self.marginals:
            x_marg = x.getMarginal(f.getDescription())
            grad_f = f.computeLogPDFGradient(x_marg)

            # handles truncated distributions
            if "TruncatedDistribution" in str(f.__class__.__name__):
                grad_blocks.append(grad_f.getMarginal([0, 1]))
            else:
                grad_blocks.append(grad_f)
            
        if stack:
            return JointDistribution.merge_samples(grad_blocks)
        else:
            return grad_blocks


    def computePDFGradient(self, x: ot.Sample, stack=True):
        grad_blocks = []

        for f in self.marginals:
            x_marg = x.getMarginal(f.getDescription())
            grad_f = f.computePDFGradient(x_marg)

            # handles truncated distributions
            if "TruncatedDistribution" in str(f.__class__.__name__):
                grad_blocks.append(grad_f.getMarginal([0, 1]))
            else:
                grad_blocks.append(grad_f)
            
        if stack:
            return JointDistribution.merge_samples(grad_blocks)
        else:
            return grad_blocks
        

    def fisherInformation(self, stack=True):

        information_blocks = []
        for f in self.marginals:
            information_blocks.append(f.fisherInformation())
        
        if stack:
            # build the full numpy matrix from the diagonal blocks
            FIM = scipy.linalg.block_diag(*information_blocks)
            return FIM
        else:
            return information_blocks

    def exponentialMap(self, v):
        """
        exponential map for JointDistribution class
        
        v: list of tangent vectors on each marginal (more complicated in case of dependence, TODO)
        """
        marginal_exp_map = []

        # parallelize ?
        for i, f in enumerate(self.marginals):
            g = f.exponentialMap(v[i])
            desc = f.getDescription()
            g.setDescription(desc)
            marginal_exp_map.append(g)

        joint = JointDistribution(marginal_exp_map, ot.IndependentCopula(len(self.marginals)))
        joint.setDescription(self.getDescription())
        
        return joint
    
if __name__ == "__main__":

    # g = ExtendedTrNormal(0, -1, 2, 3)
    # g.setDescription('X1')
    g = ExtendedTrNormal(0, -1, -1, 1)
    g.setDescription(ot.Description(['X1']))

    h = Normal(1, 2)
    h.setDescription(ot.Description(['X2']))

    marginals = [g, h]
    copula = ot.IndependentCopula(2)

    f = JointDistribution(marginals, copula)
    s = f.getSample(1)
    print("sample desc=", s.getDescription())
    j = f.computeLogPDFGradient(s, stack=False)
    print(j[0], j[1])
    
