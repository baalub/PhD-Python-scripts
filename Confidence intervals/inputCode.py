import openturns as ot
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.optimize import minimize

# import local modules from the same directory
from likelihood import Likelihood 

# import modules from another directory

from pathlib import Path

# get the current directory
current_dir = Path.cwd()

# move back
parent_dir = current_dir.parent
file_path = parent_dir / "Multivariate_case"

sys.path.append(str(file_path))


from truncatedDistribution import TruncatedDistribution  
from normal import Normal
from lognormal import LogNormal
from extendedtrnormal import ExtendedTrNormal
from extendedtrlognormal import ExtendedTrLogNormal
from jointdistribution import JointDistribution

class SimulationCodeInputs:
    """
    Class for inputs of computer simulation codes which allows to store data on the 
    inputs (the distribution of each input, their names and the truncation bounds). 
    The methods in this class allow to compute the perturbed distributions of each
    inputs, the quantile on each perturbed inputs and the quantile NACIs.
    
    This class is specific to the cathare simulation code, although a few modifications 
    can be made to do robustness studies on other computer simulation codes.
    """

    @staticmethod
    def listToInterval(inputVariables, trBounds):
        trBoundsIntervals = {}
        for input in inputVariables:
            a, b = trBounds[input][0], trBounds[input][1]
            trBoundsIntervals[input] = ot.Interval(a, b)
        return trBoundsIntervals
    

    def __init__(self, inputVariables, inputNames, distTypeStr, truncationBounds,
                 Dependence = False, copula = None, distribution = None, X_sample = None
                 ):
        
        self.inputVariables = inputVariables
        self.inputNames = inputNames
        self.distTypeStr = distTypeStr
        self.truncationBounds = truncationBounds

        self.Dependence = Dependence
        self.copula = copula
        self.X_sample = X_sample
        self.distribution = distribution

    @classmethod
    def from_dataframe(cls, df):
        inputNames = list(df['name'])
        inputVariables = list(df['input_variable'])

        distTypeStr = {}
        truncationBounds = {}

        for i, name in enumerate(inputVariables):
            distTypeStr[name] = df['distribution_type'][i]
            truncationBounds[name] = [ df['min'][i], df['max'][i] ]

        return cls(inputVariables, inputNames, distTypeStr, truncationBounds)
    
    
    # method that allows to determine the marginals of the input distribution
    # this method allows us to distinguish, for instance, the Normal and the 
    # ot.Distribution(ot.Normal) distributions (this is important for the Fisher Rao
    # based input perturbation method)
    def _inputMarginals(self, variables, extended):
        dictDistTypes = self.distTypeStr
        trBounds = self.truncationBounds
        marginals = {}

        for input in variables:
            lb, ub = trBounds[input][0], trBounds[input][1]

            if dictDistTypes[input] == "Normal":
                interval = ot.Interval(lb, ub)
                mu = (lb + ub)/2
                sigma = (ub - mu)/3

                if extended == True:
                    k, s = mu/sigma**2, 1/sigma**2
                    marginal = ExtendedTrNormal(k, s, lb, ub)
                else:
                    N = Normal(mu, sigma)
                    marginal = TruncatedDistribution(N, interval)
                
                marginal.setDescription(ot.Description([input]))
                marginals[input] = marginal
        
            if dictDistTypes[input] == "LogNormal":
                interval = ot.Interval(lb, ub)
                mu = (np.log(lb) + np.log(ub))/2
                sigma = (np.log(ub) - mu)/3

                if extended == True:
                    k, s = mu/sigma**2, 1/sigma**2
                    # TODO
                    marginal = ExtendedTrLogNormal(k, s, lb, ub)
                else:    
                    logN = LogNormal(mu, sigma)
                    marginal = TruncatedDistribution(logN, interval)
                
                marginal.setDescription(ot.Description([input]))
                marginals[input] = marginal

            if dictDistTypes[input] == "Uniform":
                if extended:
                    marginal = ExtendedTrNormal(0, 0, lb, ub)
                else:
                    marginal = ot.Uniform(lb, ub)

                #print(ot.Description([input])[0])
                marginal.setDescription(ot.Description([input]))
                marginals[input] = marginal

            if dictDistTypes[input] == "LogUniform":
                if extended:
                    #TODO
                    marginal = ExtendedTrLogNormal(0, 0, lb, ub)
                else:
                    marginal = ot.LogUniform(np.log(lb), np.log(ub))    
            
                marginal.setDescription(ot.Description([input]))
                marginals[input] = marginal
        
        return marginals

    # the purpose of the following method is to convert the ot distributions to our
    # distributions (important for FisherRao computations)
    def _convertInputMarginals(self):
        marginals = self._inputMarginals()
        distTypeStr = self.distTypeStr
        trBounds = SimulationCodeInputs.listToInterval(self.inputVariables, self.truncationBounds)
        dictConvertedMarginals = {}

        for input in self.inputVariables:
            marginal = marginals[input]
            m, s = marginal.getParameter()[0], marginal.getParameter()[1]
            nameStr = distTypeStr[input]

            # get the module whose name is the class's name (in small letter).
            # By convention, we named all modules containing our classes
            # using the classes' names in Camel case
            mod = sys.modules[nameStr.lower()]
            
            # get the class contained in the module 
            cls = getattr(mod, nameStr)

            # define the converted marginals
            convMarginal = TruncatedDistribution(cls(m, s), trBounds[input])
            dictConvertedMarginals[input] = convMarginal

        return dictConvertedMarginals

    # method for building the input distributions of the simulation code
    # well adapted to truncated distributions
    def buildInputDistribution(self, extended=False):
        """
        build the initial distribution on the vector of inputs. The marginals 
        are assumed independent.
        """
        marginals = self._inputMarginals(self.inputVariables, extended=extended)
        marginals = list(marginals.values())
        copula = ot.IndependentCopula(len(marginals))
        dist = JointDistribution(marginals, copula)
        # desc = ot.Description([f.getDescription()[0] for f in marginals])
        # dist.setDescription(desc)
        return dist

    def buildPartialDistribution(self, perturb_inputs, copula, extended=False):
        """
        build the partial distribution on the vector of inputs to be perturbed.
        """
        marginals = self._inputMarginals(perturb_inputs, extended=extended)
        marginals = list(marginals.values())
        dist = JointDistribution(marginals, copula)

        return dist

    # set the input distribution to the object
    def setInputDistribution(self, inputDistribution):
        self.distribution = inputDistribution
    
    def getInputDistribution(self):
        if self.distribution == None:
            print("input distribution not defined yet")
        else:
            return self.distribution
        
    # method for attributing the input sample to the object
    # it is used as a seperate method to allow the user to first
    # define the inputs (name, variable number, distribution type)
    # and then add the input sample later
    def setInputSample(self, inputSample):
        self.X_sample = inputSample

    def _buildPerturbedDistributionsFisherRao(self, amount, delta):
        trBounds = self.truncationBounds

        # convert the interval to an actual OpenTURNS Interval (important for FisherRao computations)
        trBoundsIntervals = SimulationCodeInputs.listToInterval(self.inputVariables, trBounds)

        # get the marginals (converted to our classes)
        marginals = self._convertInputMarginals()
        dictPerturbedDist = {}
        
        for input in self.inputVariables:
            dictPerturbedDist[input] = marginals[input].sampleFisherRaoSphere(delta, amount, trBoundsIntervals[input])

        return dictPerturbedDist

    # method for building the perturbed versions of the input distributions
    def buildPerturbedDistributions(self, amount, FisherRao = False, delta = 0.5):
        dictDistTypes = self.distTypeStr
        trBounds = self.truncationBounds
        dictPerturbedDist = {}
        
        if FisherRao:
            return self._buildPerturbedDistributionsFisherRao(amount, delta)
        
        # we begin objectiterating on the list of input variables' names
        for input in self.inputVariables:
            lowerTrunc, upperTrunc = trBounds[input][0], trBounds[input][1]
            interval = ot.Interval(lowerTrunc, upperTrunc)
            perturbedList = []

            if dictDistTypes[input] == "Normal":
                m = (lowerTrunc + upperTrunc)/2
                s = (upperTrunc - m)/3

                for i in range(amount):
                    u = np.random.uniform(m-1, m+1)
                    v = np.random.uniform(s-0.2, s+0.5)
                    while v<= 0:
                        v = np.random.uniform(s-0.2, s+0.5)

                    n = ot.Normal(u, v)
                    pert =  ot.TruncatedDistribution(n, interval)
                    perturbedList.append(pert)                    

                dictPerturbedDist[input] = perturbedList
            
            if dictDistTypes[input] == "LogNormal":
                m = (np.log(lowerTrunc) + np.log(upperTrunc))/2
                s = (np.log(upperTrunc) - m)/3

                for i in range(amount):
                    u = np.random.uniform(m-1, m+1)
                    v = np.random.uniform(s-0.2, s+0.5)
                    while v<= 0:
                        v = np.random.uniform(s-0.2, s+0.5)

                    ln = ot.LogNormal(u,v)
                    pert = ot.TruncatedDistribution(ln, interval)
                    perturbedList.append(pert)

                dictPerturbedDist[input] = perturbedList
            
        return dictPerturbedDist

    # method based on the empiricalWeightedQuantile method from the 
    # Likelihood class
    def estimatePerturbedQuantiles(
            self, perturbedInputs, alpha, dictPerturbedDistributions, X_sample, Y_sample
            ):
        
        inputDist = self.distribution
        dictEstimatedQuantiles = {}

        # we begin iterating on the list of input variables' names
        for j,input in enumerate(perturbedInputs):
            # amount = len(dictPerturbedDistributions[input])
            listPerturbedQuantiles = []
            listPerturbedDists = dictPerturbedDistributions[input]
            #distribution
            marginal = inputDist.getMarginal(j)
            #print(marginal.getDescription())

            for f_pert in listPerturbedDists:
                pertLikelihood = Likelihood(f_pert, marginal, pushforward=True)
                q = pertLikelihood.empiricalWeightedQuantile(
                    Y_sample=Y_sample, alpha=alpha, X_sample=X_sample[input]
                )
                listPerturbedQuantiles.append(q)

            dictEstimatedQuantiles[input] = listPerturbedQuantiles

        return dictEstimatedQuantiles
    
    # basic method for computing the perturbed distribution which gives the 
    # highest estimated quantile
    def computeMaxEstimation(self, dictPerturbedDists, dictEstimatedQuantiles):
        """
        This method allows to compute the worst case quantile (i.e. maximum) for a given list
        of perturbed quantile estimation (for each considered input)
        """
        dictMaxDist = {}
        for input in self.inputVariables:
            maxIndex = np.argmax(dictEstimatedQuantiles[input])
            maxDist = dictPerturbedDists[input][maxIndex]
            dictMaxDist[input] = maxDist

        return dictMaxDist
    
    # method also based on the Likelihood class
    def buildPerturbedLikelihoods(self, perturbedDistributions):
        f0 = self.buildNominalDistribution()
        likelihoodsList = []

        for f in perturbedDistributions:
            lowerBound, upperBound, varianceBound = self.computeLikelihoodBounds(f)
            l = Likelihood(f, f0, lowerBound, upperBound, varianceBound, pushforward=True)
            likelihoodsList.append(l)

        return likelihoodsList

    # builds the non-asymptotic confidence bounds for output perturbed quantiles   
    def quantileNACIs(
            self, perturbedInputs, perturbedDistributions, X_sample, Y_sample, alpha,
            beta = 0.95, intervalType = "right bound", CiBoundType="Hoeffding"
    ):
        """
        Method which computes the non-asymptotic confidence intervals for perturbed quantiles.

        perturbedDistributions: a dictionnary where the keys are the input variable names and the value
        is a perturbed distribution of the nominal distribution of this input variable
        Y_sample: the output sample
        alpha: quantile order of the output for which we are building the CIs
        beta: the confidence level, default value is 0.95 (non-asymptotic)
        intervalType: either "left bound", "right bound" or "two-sided"
        CiBoundType: either "Hoeffding" or "Bennett"
        """

        trBounds = self.truncationBounds
        inputDist = self.distribution
        DictNACIs = {}

        # iterate on the number of perturbed variables
        for j, input in enumerate(perturbedInputs):

            # build the likelihood ratio between the marginal of input and the perturbed dist
            pertLikelihood = self._dictLikelihoodWithBounds(perturbedInputs, perturbedDistributions)        
            # compute the NACI for this likelihood using the computeCI method from the Likelihood
            # class
            DictNACIs[input] = pertLikelihood[input].computeCI(
                Y_sample, alpha, beta, X_sample=X_sample[input], intervalType=intervalType, CiBoundType=CiBoundType
            )
        
        return DictNACIs
    
    def _dictLikelihoodWithBounds(self, perturbedInputs, perturbedDistributions):
        inputDist = self.distribution
        trBounds = self.truncationBounds

        dictLWithBounds = {}

        for j, input in enumerate(perturbedInputs):
            # build the likelihood ratio between the marginal of input and the perturbed dist
            pertLikelihood = Likelihood(
                mu=perturbedDistributions, mu0= inputDist.getMarginal(j), pushforward=True
            )

            # compute the bounds on the likelihood ratio using the computeLikelihoodBounds method
            # from the Likelihood class and then attribute the bounds to the likelihood object
            likelihoodBounds = pertLikelihood.computeLikelihoodBounds(
                truncated = True, truncationBounds=trBounds[input]
            )
            pertLikelihood.setLikelihoodBounds(likelihoodBounds)
            dictLWithBounds[input] = pertLikelihood

        return dictLWithBounds
    
    # method which prints the quantile CIs
    def printQuantileCIs(self, dictNACIs, alpha):
        
        inputVars = self.inputVariables
        
        for j, input in enumerate(self.inputVariables):
            CIs = dictNACIs[input]
            print(f"perturbation of {input} ({self.inputNames[j]}): {int(alpha*100)}%-quantile is in [{CIs[0]},{CIs[1]}] with {CIs[2]*100}% probability")


    # method for plotting the perturbed distributions using matplotlib
    def plotPerturbedDistributions(self, dictPerturbedDist, dictWorstCaseDist):
        inputVars = self.inputVariables 
        inputDist = self.distribution

        # for each input, create a subplot
        n = len(inputVars)
        r = n%3
        if r ==0:
            d = int(n/3)
        else:
            d = int(n/3) +1

        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(d, 3, figsize=(16, 9))
        ax = ax.flatten()

        # iterate on the number of perturbed inputs
        for i, input in enumerate(inputVars):
            lbT, ubT = self.truncationBounds[input][0], self.truncationBounds[input][1]
            f_i = inputDist.getMarginal(i) # nominal marginal distribution
            f_max = dictWorstCaseDist[input]

            amount = len(dictPerturbedDist[input])
            e = (ubT -lbT)*0.05
            X = np.linspace(lbT-e, ubT+e, 500)
            # plot the nominal distribution
            ax[i].plot(X, [f_i.computePDF(x) for x in X], lw=1.8, color = "blue", label="nominal")
        
            for j in range(amount):       
                # plot the perturbed distributions
                f_th = dictPerturbedDist[input][j]
                ax[i].plot(X, [f_th.computePDF(x) for x in X], lw =1.2, color = "red", label='_nolenged_', alpha = 0.25)
                
            # plot worst case distribution
            ax[i].plot(X, [f_max.computePDF(x) for x in X], lw =1.8, color = "red", label = "worst case")
            ax[i].tick_params(axis='x', labelsize=14)
            ax[i].tick_params(axis='y', labelsize=14)

            ax[i].set_title(f"{self.inputNames[i]}")

            #dummy plot for legend
            ax[i].plot([], [], color = "red", label='perturbed', alpha = 0.3)            

            ax[i].legend(fontsize=14)
            ax[i].grid(True)

        if r != 0:
            ax[int(d*3-1)].axis('off') 

        fig.tight_layout()
        fig.savefig("figures/perturbed_dists.png", dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()

    def plotPerturbedLikelihoods(self, dictPerturbedDist, dictWorstCaseDist):
        inputVars = self.inputVariables 
        inputDist = self.distribution
        dictLikelihoodnBounds = self._dictLikelihoodWithBounds(dictWorstCaseDist)

        # for each input, create a subplot
        n = len(inputVars)
        r = n%3
        if r ==0:
            d = int(n/3)
        else:
            d = int(n/3) +1

        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(d, 3, figsize=(16, 9))
        ax = ax.flatten()

        for i, input in enumerate(inputVars):

            pertLikelihood = dictLikelihoodnBounds[input]
            lowerBound, upperBound = pertLikelihood.getBounds()

            lbT, ubT = self.truncationBounds[input][0], self.truncationBounds[input][1]
            f_i = inputDist.getMarginal(i) # nominal marginal distribution
            f_max = dictWorstCaseDist[input]
            amount = len(dictPerturbedDist[input])
            X = np.linspace(lbT+0.0001, ubT-0.0001, 500)

            for j in range(amount):
                # plot the likelihood btw perturbed and nominal
                f_th = dictPerturbedDist[input][j]
                ax[i].plot(X, [f_th.computePDF(x)/f_i.computePDF(x) for x in X], color = "blue", label='_nolenged_')
                
            # plot worst case likelihood along with the bounds
            ax[i].plot(X, [f_max.computePDF(x)/f_i.computePDF(x) for x in X], color = "red", label = "worst case")
            ax[i].plot([lbT, ubT], [lowerBound, lowerBound], color = 'orange', label = 'likelihood min value')
            ax[i].plot([lbT, ubT], [upperBound, upperBound], color = 'orange', label = 'likelihood max value')
            ax[i].tick_params(axis='x', labelsize=12)
            ax[i].tick_params(axis='y', labelsize=12)
            ax[i].set_title(f"{self.inputNames[i]}")

            #dummy plot for legend
            ax[i].plot([], [], color = "blue", label='perturbed')            

            ax[i].legend()
            ax[i].grid(True)

        if r != 0:
            ax[int(d*3-1)].axis('off') 

        fig.tight_layout()
        fig.savefig("figures/perturbed_likelihoods.png", dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # plot the min and max for each perturbed likelihood to show that 
        # the scipy.optimize method has well converged

###---------------------Previous versions----------------------------###

# def computeLikelihoodBounds(self, perturbedDistribution):
#         f0 = self.buildInputDistribution()
#         f_th = perturbedDistribution

#         # get the lower and upper truncation bounds and define the starting point x0
#         # for the optimization solver

#         lbT, ubT = self.truncationBounds, self.truncationBounds
#         x0 = np.array(f_th.computeQuantile(0.5))[0] 

#         lowerBound = minimize(lambda x: f_th.computePDF(x)/f0.computePDF(x), x0=x0,
#                               bounds=[(lbT, ubT)] ).fun
        
#         upperBound = minimize(lambda x: -f_th.computePDF(x)/f0.computePDF(x), x0=x0,
#                               bounds=[(lbT, ubT)]).fun*(-1)
        
#         varianceBound = scp.integrate.quad(lambda x: f_th.computePDF(x)**2/f0.computePDF(x),
#                                            lbT, ubT)[0]
        
#         print("min value =", lowerBound)
#         print("max value =",upperBound)
#         print("variance =", varianceBound)

#         return lowerBound, upperBound, varianceBound
