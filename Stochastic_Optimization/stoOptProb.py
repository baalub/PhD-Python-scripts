import openturns as ot
import numpy as np
import pandas as pd
import time
import sys
import os
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
import random
from pathlib import Path


ot.RandomGenerator.SetSeed(random.randrange(1, 2**31 -1))

# import modules from another directory

# get the current directory
current_dir = Path.cwd()

# move back
parent_dir = current_dir.parent
file_path_1 = parent_dir / "Multivariate_case"

sys.path.append(str(file_path_1))

from truncatedDistribution import TruncatedDistribution
from jointdistribution import JointDistribution

from normal import Normal
from lognormal import LogNormal
from extendedtrnormal import ExtendedTrNormal

class RiemannianStochasticOptimization:
    """
    Stochastic optimization on Riemannian manifolds (parametric families) usually
    involving objective functions given by an expectation/quantile on the output 
    of a black-box function.
    
    Parameters
    ----------
    objectiveFunction : ot.PythonFunction
        The black-box function to be optimized/on which distributional robustness is performed.

    fullDistribution : JointDistribution
        The initial probability distribution on the input space. The marginals should have an
        ot.Description before the JointDistribution is built. Note that ot.JointDistribution
        is not accepted since does not contain the exponentialMap and fisherInformation methods.

    inputDesc : list 
        The list of input variable descriptions.

    perturbedInputs : list
        The list of strings representing the variables to be perturbed.
    
    partialDistribution : JointDistribution
        The initial probability distribution on the marginals that will be perturbed

    """

    # @staticmethod
    # def H(y, alpha, q):
    #     return q + 1/(1-alpha) * np.max(y-q,0)
    
    @staticmethod
    def H(y, alpha, q):
        return y
    
    @staticmethod
    def partial_phi(y, alpha, q):
        y = np.array(y).transpose()[0]
        return 1 + 1/(1-alpha) *(y>q)
    
    @staticmethod
    def phi(y, alpha, q):
        y = np.array(y).transpose()[0]
        return q + 1/(1-alpha) * (y - q)*(y>q)

    @staticmethod
    def IGO_Q(alpha: float, array: np.ndarray):

        N = len(array)
        arange = np.arange(1, N+1)
        arg = (arange + np.ones(N)/2)/N
        omegas = arg >= alpha
        sorted_index = np.argsort(np.argsort(array))
        #print("weights =", omegas[sorted_index])
        return omegas[sorted_index]

    @staticmethod
    def CMA_ES(lamb, array):
        N = len(array)
        sorted_index = np.argsort(-array)
        arange = np.arange(1, N+1)
        num = [max(0, np.log(lamb/2 +1) - np.log(i)) for i in arange]
        den = np.sum(num)

        util = num/den - 1/lamb*np.ones(N)
        return util[np.argsort(sorted_index)]

    @staticmethod
    def identity(lamb, array):
        return array

    def __init__(
            self, objectiveFunction, fullDistribution: JointDistribution, 
            inputDesc, perturbedInputs, partialDistribution=None #, retraction=None
            ):

        self.objectiveFunction = objectiveFunction # should always be wrapped as an ot function
        self.fullDistribution = fullDistribution # OpenTURNS JointDistribution

        if partialDistribution != None:
            self.partialDistribution = partialDistribution
        else:
            self.partialDistribution = fullDistribution

        self.perturbedInputs = perturbedInputs
        self.inputDesc = inputDesc

        # TODO: fix the description problem here
        
        # self.retraction = retraction
        

    def _buildFullPerturbedDistribution(self, partialDistribution: JointDistribution):
        """
        This method allows to build the full distribution from the 
        perturbed marginal distributions and the initial full distribution.
        """
        # the list of strings of input variables whose distribution is perturbed
        # for example perturbedInputs = ["X1", "X38", "X54"] and f_k is a distribtution 
        # on (X1, X38, X54)
        perturbedInputs = self.perturbedInputs
        initialFullDistribution = self.fullDistribution
        inputDesc = self.inputDesc

        marginals = {}
        j = 0 

        for i, input in enumerate(inputDesc):
            if input not in perturbedInputs:
                # print(partialDistribution.getDimension())

                f = initialFullDistribution.marginals[i]
                f.setDescription(ot.Description([input]))
                marginals[input] = f
                
            else:
                f = partialDistribution.getMarginal(j)
                f.setDescription(ot.Description([input]))
                marginals[input] = f
                j+=1

        margs = list(marginals.values())
        finalFullDistribution = JointDistribution(margs, ot.IndependentCopula(len(inputDesc)))
        finalFullDistribution.setDescription(ot.Description(inputDesc))

        return finalFullDistribution

        # # update the perturbed marginals (remembering python's indexing convention)
        # for i, input in enumerate(perturbedInputs):
        #     marginals[int(input[1:])-1] = partial_dist.getSingleMarginal(i)

        # copula = full_dist_0.getCopula()
        # perturbedDistribution = ot.JointDistribution(marginals, copula)
        
        # return perturbedDistribution


    def naturalEvolutionStrategy(
            self, fitnessShapingStr = "IGO_Q", eta=3,
            nbIter=100, n=50, fitnessParameter=5,
            sampleReuse=False, recycleDepth=5,
            mixtureWeights=None, recursiveSaving=False, base_dir=None,
            # Y_samples=[], iterations=[], fullIterations=[]
            ):
        """
        Approximates the region where the value of the function is maximal. 
        
        Parameters
        ----------
        fitnessShapingStr : str
            The type of weights that are used at the fitness shaping step. For quantile
            optimization, 'IGO_Q' should be picked and for general maximization of the objective function, 
            'CMA_ES' can be chosen.

        fitnessParameter : float or list
            if 'IGO_Q' is chosen, then fitnessParameter corresponds to the quantile order which can depend
            on the iteration.

        eta : float
            the step size in the gradient ascent (may depend on nbIter, but not taken into account here).

        nbIter : int
            the number of total iterations of the algorithm

        n : int
            the sample size at each iteration (may depend on nbIter, but not taken into account here)

        sampleReuse : bool
            if the previous samples are recycled or not (only for 'IGO_Q')

        recycleDepth : int
            how far back from the current iteration are the samples recycled (may depend on nbIter)
        
        mixtureWeights : np.ndarray
            if sampleReuse==True, the weights for the mixture distributions of the last recycleDepth iterations.
            To give to ot.Mixture() as weights. 
        
        Returns
        -------
        tuple of list
            The iterations of the partial and full distributions produced, the input and output samples
            used and the specified parameter arguments.

        This algorithm is based on the paper "Natural Evolution Strategies" by D. Wierstra et al., 2014 and "Information-Geometric 
        Optimization: a unifying picture through invariance principles" by Y. Ollivier et al, 2017. Also possible to search for 
        regions partially by fixing the distribution of all but a few marginals.

        Note that all distributions (even unidimensional) should be instances of the JointDistribution class (and not 
        ot.JointDistribution) !
        """

        # build the dictionnary of input variables and values, useful for plotting later
        kwargs_inputs = {'fitnessParameter': [fitnessParameter], 'fitnessShapingStr': [fitnessShapingStr],
                        'nbIter': [nbIter], 'eta': [eta], 'n': [n], 'sampleReuse': [sampleReuse], 
                        'recycleDepth': [recycleDepth]}


        partialDistribution = self.partialDistribution
        fullDistribution = self.fullDistribution

        # objective function
        h = self.objectiveFunction

        # which method for the fitness shaping step
        fitnessShaping = getattr(RiemannianStochasticOptimization, fitnessShapingStr)

        #initialize empty lists for storing the iterations, 
        # input and output samples (later used for emp. quantile and emp. average computation)
        iterations = [] # list for the iterations on the partial distributions
        fullIterations = [] # list for the iterations on the full distributions (important for sample reuse)
        X_samples = []
        Y_samples = []

        iterations.append(partialDistribution)
        fullIterations.append(fullDistribution)

        sample_size = n
        start = len(iterations)
        for k in range(start, start + nbIter+1):
            
            n = sample_size
            # checks if fitnessParameter is a list of (increasing) fitness values, allowing to apply IGO-Q  
            # with a varying parameter alpha
            if type(fitnessParameter) == list or type(fitnessParameter)==np.ndarray:
                alpha = fitnessParameter[k-1]
            else:
                alpha = fitnessParameter
            
            # compute the Fisher information at the current partial distribution
            FIM = partialDistribution.fisherInformation(stack=False)

            # sample from current full distribution
            sample = fullDistribution.getSample(n)

            # evaluate the sample on the function (either in series or in parallel, 
            # here in parallel thanks to othpc library) and append to a list
            print("computing output values...\n")
            Hvalues = h(sample)
            print(Hvalues)

            # check for nan and remove them, note that the initial sample size 
            # is overwritten by the actual output size
            sample, Hvalues, n = RiemannianStochasticOptimization._remove_nan(sample, Hvalues)

            # append the input and output samples
            X_samples.append(sample)
            Y_samples.append(Hvalues)
                        
            # to keep memory free when computing on hpc
            # print("deleting my_results folder...")
            # shutil.rmtree(f"{base_dir}/my_results")
            # os.mkdir(f"{base_dir}/my_results")
            print("output values =\n ", Hvalues)

            # if Sample reuse is desired
            if sampleReuse==True:

                # in the first case the iteration is still not larger than the recyling depth so
                # the depth is taken to be k
                if k <= recycleDepth:
                    weightedPartialGrads=RiemannianStochasticOptimization._sampleReuseStrategy(
                        iterations, fullIterations, X_samples, Y_samples, k,
                        alpha, FIM, eta, n, mixtureWeights
                        )
                    print("")
                    print("the (natural) gradient update = \n", weightedPartialGrads)

                else:
                    weightedPartialGrads=RiemannianStochasticOptimization._sampleReuseStrategy(
                        iterations, fullIterations, X_samples, Y_samples, recycleDepth,
                        alpha, FIM, eta, n, mixtureWeights
                        )
                    print("")
                    print("the (natural) gradient update = \n", weightedPartialGrads)

            else:
                # fitness shaping
                weights = fitnessShaping(alpha, np.array(Hvalues).transpose()[0])

                # compute the log pdf gradient on the sample, returns a list of the partial gradient for each perturbed marginal
                grads = partialDistribution.computeLogPDFGradient(sample, stack=False)

                # compute the estimated (natural) gradient (ie weighted grad log likelihoods) 
                weightedPartialGrads = RiemannianStochasticOptimization.combineWeightAndGradient(weights, FIM, grads, eta, n)
                print("the (natural) gradient update = \n", weightedPartialGrads)

            # compute the next iteration using the exponential map at the partial distribution
            print("computing the geodesic and updating the distribution...\n")
            partialDistribution = partialDistribution.exponentialMap(weightedPartialGrads)
            partialDistribution.setDescription(self.perturbedInputs)
            
            # store the k-th iteration (the partial distribution)
            iterations.append(partialDistribution)

            # update the full distribution after updating the partial dist 
            fullDistribution = self._buildFullPerturbedDistribution(partialDistribution) 

            # and store it in a list
            fullIterations.append(fullDistribution)

            # logging
            if k%1 == 0:
                print(f"At iteration {k}:")
                # print(f"Parameter of the current distribution: {fullDistribution.getParameter()}")
                print(f"Average value of the sample output: {np.mean(Hvalues)}")
                print(f"{50}%-quantile of the sample output: {np.quantile(Hvalues, 0.50)}")
                print(f"{95}%-quantile of the sample output: {np.quantile(Hvalues, 0.95)}")                
                print("")

            # if recursive saving is True, then recursively save the iterations
            #self.store_optimization_results(iterations, Y_samples, kwargs_inputs, base_dir, zip=True)
        
        return iterations, X_samples, Y_samples, self.perturbedInputs, kwargs_inputs

    @staticmethod
    def _remove_nan(X_sample: ot.Sample, Y_sample: ot.Sample):
        """
        Helper function for removing the lines containing 'nan' in both input and output sample.
        Useful when expensive computer simulations last longer than expected        
        """
        Y_arr = np.array(Y_sample).transpose()[0]
        # loop over the output sample and check for a nan
        for i, boolean in enumerate(np.isnan(Y_arr)):
            # if nan, erase the whole line for both input and output sample
            if boolean:
                Y_sample.erase(i)
                X_sample.erase(i)

        n_output_size = np.sum(~np.isnan(Y_arr))

        return X_sample, Y_sample, n_output_size 


    @staticmethod
    def _VstackSamples(samples: list, horizontalArray=False):
        # stack the samples using numpy
        arr = np.vstack([np.array(s) for s in samples])

        if horizontalArray == True:
            return arr.transpose()[0]
        
        desc = samples[0].getDescription()
        ot_sample = ot.Sample(arr) 
        ot_sample.setDescription(desc)

        return ot_sample
    
    @staticmethod
    def _sampleReuseStrategy(iterations: list, fullIterations: list, X_samples:list, Y_samples:list, r: float, fitnessParameter,
                             FIM, eta: float, n: int, mixtureWeights: np.ndarray):
        """
        Hidden function that handles the sample reuse strategy. It allows to recycle older samples using an importance
        mixing technique. For more details see the paper 'Sample Reuse via Importance Sampling in Information Geometric 
        Optimization' by Shirikawa et al., 2018.

        Parameters
        ----------
        iterations : list
            list of all partial distributions currently produced by the algorithm

        fullIterations : list
            list of all full distributions produced by the algorithm

        X_samples : list
            list of all input samples of type ot.Sample

        Y_samples : list 
            list of all output samples (ot.Sample)

        r : float
            the recycling depth (how far back it is recycled)

        fitnessParameter : float
            the parameter in the fitness shaping step (the quantile order for the G-robustness problem)

        FIM : list
            list of all Fisher information blocks computed at the current iteration as of type np.ndarray

        eta : float
            step size aka learning rate

        n : int
            the (fixed) sample size at each iteration

        Returns
        -------
        the list of partial weighted gradients to be given to the exponentialMap method in the JointDistribution class
        """

        # first stack vertically the the last r and current input and output samples 
        v_stacked_X_samples = RiemannianStochasticOptimization._VstackSamples(X_samples[-r:])
        v_stacked_Y_samples = RiemannianStochasticOptimization._VstackSamples(Y_samples[-r:], horizontalArray=True)

        # we need to compute three components for the gradient update: the empirical weights, the likelihood ratio btw
        # the current and the mixture distribution and the score function of current iteration. All should be evaluated 
        # on the stacked X_samples

        # first, the likelihood ratio of current wrt mixture
            # build the mixture distribution
        mixtureElements = fullIterations[-r:]

        # specify equal weight if not specified as arg
        # if mixtureWeights == None:
        #     mixtureWeights = np.ones(len(mixtureElements))

        mixture = ot.Mixture(mixtureElements)
            # extract the last partial and full iteration
        partialDistribution = iterations[-1]
        fullDistribution = fullIterations[-1]
            # compute the ratio on the stacked X_sample (parallelize this step ?)
        likelihoodRatios = np.array([fullDistribution.computePDF(x)/mixture.computePDF(x) for x in v_stacked_X_samples])


        # second, the score function of the partial distribution
                # compute the score function of the current iteration on the stacked sample
        grads = partialDistribution.computeLogPDFGradient(v_stacked_X_samples, stack=False)


        # lastly, compute the empirical weights
            # define the matrix of indicator functions of events G(X_i) < G(X_j)
        indicatorMatrix = RiemannianStochasticOptimization.build_indicator_matrix(v_stacked_Y_samples) 
            # compute the empirical probabilities
        empiricalProba = np.sum(indicatorMatrix*likelihoodRatios, axis=1)/(n*(r+1))
            # apply the selection scheme 
        weights = empiricalProba >= fitnessParameter

        # compute the total weighted gradient (to give to partialDistribution.exponentialMap())
        # the helper function below returns a list of all the partial graidents
        prod = weights*likelihoodRatios
        return RiemannianStochasticOptimization.combineWeightAndGradient(prod, FIM, grads, eta, n*(r+1))


    @staticmethod
    def build_indicator_matrix(Y: np.ndarray):
        n = len(Y)
        mat = np.zeros((n, n))
        for i, y in enumerate(Y):
            mat[i, :] = Y <= y

        return mat

    # def indicator(Y: np.ndarray):
    #     """
    #     Helper function for computing the weights in the sample reuse version of the IGO/NES algorithms
    #     """
    #     def func(y: float):
    #         return Y<=y
        
    #     return func

    @staticmethod
    def combineWeightAndGradient(weights: np.ndarray, FIM, partialGrads: list, eta, n):
        """
        helper function for computing the estimated gradient 
        
        weights: the fitness shaping weights are precomputed before this step

        returns a list containing the weights gradients on each marginal of the joint distribution
        (assumed independent)
        """
        weightedPartialGrads = []

        for i, partialGrad in enumerate(partialGrads):
            # compute the matrix product between the scores (grad log pdf) and the weights 
            weightedPartialGrad = eta*np.dot(weights.transpose(), np.array(partialGrad))/n
            # print("weighted grad=", weightedPartialGrad)
            # compute the inverse FIM
            J = np.linalg.inv(FIM[i])
            # append the natural weighted gradient
            weightedPartialGrads.append(J@weightedPartialGrad)
        
        return weightedPartialGrads

    @staticmethod
    def _increasing_alpha(nbIter: float, which_type: str):
        if which_type == "Bardou et al.":
            Alpha = list(0.5*np.ones(int(nbIter/3)+1)) + list(0.75*np.ones(int(nbIter/3))) + list(0.95*np.ones(int(nbIter/3)+1))
            return Alpha

    def store_optimization_results(
            self, iterations: list, Y_samples, kwargs_inputs: dict, base_dir, zip: bool=True) -> pd.DataFrame:
        """
        This method allows to store in a dataframe all the results from the optimization
        algorithm. This will be useful when the algorithm is performed on HPC for a costly
        objective function, allowing to store the optimization results for postprocessing.
        """

        # time tag
        time_tag = datetime.now().strftime("%d-%m-%Y_%H-%M")

        # creating the dataframe with the Y samples
        Y_samples_arr = np.vstack([np.array(y).transpose()[0] for y in Y_samples])
        df_Ysamples = pd.DataFrame(Y_samples_arr)

        # convert the kwargs_inputs dictionnary into a dataframe and add the time_tag
        kwargs_inputs["time_tag"] = [time_tag]
        df_kwargs_inputs = pd.DataFrame.from_dict(kwargs_inputs)

        # and for the iterations
        N = len(iterations)
        eff_dim = iterations[0].getDimension()

        # the columns of the perturbed inputs and the 
        column_names = self.perturbedInputs 

        # define the dataframe with the headers
        df_iterations = pd.DataFrame(columns=column_names)

        # store the class names of the marginal distributions
        for i, input in enumerate(self.perturbedInputs):
            marginal_i = iterations[0].marginals[i]
            class_name = marginal_i.__class__.__name__
            
            # check if marginal_i is a truncated distribution
            if class_name == "TruncatedDistribution":
                # get the name of the base distribution
                base_dist_name = marginal_i._custom_base_distribution.__class__.__name__

                # and store it along with the TruncatedDistribution string 
                df_iterations.loc[0, input] = [class_name, base_dist_name]
            else:
                df_iterations.loc[0, input] = class_name

        # store the class name of the copula
        #copula = iterations[0].copula
        # df_iterations.loc[0, input] = copula.__class__.__name__
        
        for k, f in enumerate(iterations):
            list_param = []
            # extract the marginal parameters and store it in a list
            for marginal in f.marginals:
                # get the description input corresponding to the marginal
                desc_str = str(marginal.getDescription())[1:-1]
            
                df_iterations.loc[k+1, desc_str] = list(marginal.getParameter())
                
            # add list_param to the dataframe

            #df_iterations[f"marginal_{i+1}" for i in range(eff_dim)] = list_param
        
        # for the perutrbed inputs ?
        # df_perturbed_inputs = pd.DataFrame()
        # df_perturbed_inputs.loc[0, "perturbed inputs"] = self.perturbedInputs

        # create a folder containing all the .csv files from the previous DataFrames
        child_path = os.path.join(base_dir, "optimization_results")
        result_dir = f"result_{time_tag}"
        full_path = os.path.join(child_path, result_dir)

        os.makedirs(full_path, exist_ok=True)  # create the directory

        df_iterations.to_pickle(f"{full_path}/iterations_{time_tag}.pkl")
        df_Ysamples.to_pickle(f"{full_path}/Ysamples_{time_tag}.pkl")
        df_kwargs_inputs.to_pickle(f"{full_path}/kwargs_inputs_{time_tag}.pkl")

        # for zipping the file
        if zip == True:
            # if the .zip file exists, then delete it
            if os.path.exists(f"{child_path}/result_{time_tag}.zip"):
                os.remove(f"{child_path}/result_{time_tag}.zip")

            # zip the file
            shutil.make_archive(f"{child_path}/result_{time_tag}", 'zip', full_path)

            # delete the folder
            if os.path.exists(full_path):
                shutil.rmtree(full_path)

        else:
            return df_iterations, df_Ysamples, df_kwargs_inputs, time_tag
    


    def expectationBasedOptimization(
            self, phi: ot.PythonFunction, eta=3, nbIter=100, n=50, 
            sampleReuse=False, recycleDepth=5, mixtureWeights=None
    ):
        
        # build the dictionnary of input variables and values, useful for plotting later
        kwargs_inputs = {'nbIter': [nbIter], 'eta': [eta], 'n': [n], 'sampleReuse': [sampleReuse], 
            'recycleDepth': [recycleDepth]}

        partialDistribution = self.partialDistribution
        fullDistribution = self.fullDistribution

        # objective function
        h = self.objectiveFunction

        #initialize empty lists for storing the iterations, 
        # input and output samples (later used for emp. quantile and emp. average computation)
        iterations = [] # list for the iterations on the partial distributions
        fullIterations = [] # list for the iterations on the full distributions (important for sample reuse)
        X_samples = []
        phi_samples = []

        iterations.append(partialDistribution)
        fullIterations.append(fullDistribution)

        for k in range(1, nbIter+1):

            print("parameter iteration", partialDist.getParameter())
            # compute the Fisher information at the current partial distribution
            FIM = partialDistribution.fisherInformation(stack=False)

            print("FIM", FIM)
            # sample from current full distribution and append to a list
            sample = fullDistribution.getSample(n)
            X_samples.append(sample)

            # evaluate the sample on the function (either in series or in parallel, 
            # here in parallel thanks to othpc library) and append to a list
            print("computing output values...\n")
            Y_values = h(sample)
            phi_values = phi(Y_values)
            phi_samples.append(phi_values)
            print("output values =\n ", phi_values)

            # if Sample reuse is desired
            if sampleReuse==True:

                # in the first case the iteration is still not larger than the recyling depth so
                # the depth is taken to be k
                if k <= recycleDepth:
                    weightedPartialGrads=RiemannianStochasticOptimization._sampleReuseStrategy(
                        iterations, fullIterations, X_samples, Y_samples, k,
                        FIM, eta, n, mixtureWeights
                        )
                
                else:
                    weightedPartialGrads=RiemannianStochasticOptimization._sampleReuseStrategy(
                        iterations, fullIterations, X_samples, Y_samples, recycleDepth,
                        FIM, eta, n, mixtureWeights
                        )
                    print("")
                    print("the (natural) gradient update = \n", weightedPartialGrads)

            else:
                # compute the log pdf gradient on the sample, returns a list of the partial gradient for each perturbed marginal
                grads = partialDistribution.computeLogPDFGradient(sample, stack=False)

                # compute the estimated (natural) gradient (ie weighted grad log likelihoods) 
                weightedPartialGrads = RiemannianStochasticOptimization.combineWeightAndGradient(
                    np.array(phi_values).transpose()[0], FIM, grads, eta, n
                    )
                print("the (natural) gradient update = \n", weightedPartialGrads)

            # compute the next iteration using the exponential map at the partial distribution
            print("computing the geodesic and updating the distribution...\n")
            partialDistribution = partialDistribution.exponentialMap(weightedPartialGrads)
            partialDistribution.setDescription(self.perturbedInputs)
            
            # store the k-th iteration (the partial distribution)
            iterations.append(partialDistribution)

            # update the full distribution after updating the partial dist 
            fullDistribution = self._buildFullPerturbedDistribution(partialDistribution) 

            # and store it in a list
            fullIterations.append(fullDistribution)

            # logging
            if k%1 == 0:
                print(f"At iteration {k}:")
                # print(f"Parameter of the current distribution: {fullDistribution.getParameter()}")
                print(f"Average value of the sample output: {np.mean(phi_values)}")
                print(f"{50}%-quantile of the sample output: {np.quantile(phi_values, 0.50)}")
                print(f"{95}%-quantile of the sample output: {np.quantile(phi_values, 0.95)}")                
                print("")
            
        return iterations, X_samples, phi_samples, self.perturbedInputs, kwargs_inputs

    def superquantileSGD(
            self, sampling_dist, eta, nbIter, alpha, gamma, q0
    ):

        fullDistribution = self.fullDistribution
        fullDist = fullDistribution.marginals[0]
        h = self.objectiveFunction
        
        q = q0
        iterations = []
        quantiles = []

        iterations.append(fullDistribution)
        quantiles.append(q)
        

        for k in range(1, nbIter+1):
            print("k", k)
            # usual stochastic step size and Fisher information
            eta_k = eta*k**(-gamma)
            print("eta_k=", eta_k)
            # print("eta_k = ", eta_k)
            FIM = fullDistribution.fisherInformation(stack=False)
            # print(FIM)

            # compute the output value
            print(fullDistribution.getParameter())
            x  = sampling_dist.getSample(1)
            print(x)
            y = h(x)

            # compute the gradient
            likelihood = np.array(fullDistribution.computePDF(x))[0]/np.array(sampling_dist.computePDF(x))[0]
            # print("likelihood=", likelihood)
            v_theta = self.phi(y, alpha, q)*fullDistribution.computeLogPDFGradient(x)*likelihood
            v_q = self.partial_phi(y, alpha, q)*likelihood
            
            # print("v_theta=", v_theta)
            # compute the updates for both quantile and parameter
            q = q - eta_k*v_q
            fullDistribution = fullDistribution.exponentialMap(self._compute_natural_grad(FIM, v_theta, -eta_k))
            
            # print("theta=", fullDistribution.getParameter())
            # print("q=", q)

            iterations.append(fullDistribution)
            quantiles.append(float(q))
        
        return iterations, quantiles
    
    @staticmethod
    def _compute_natural_grad(mats, vecs, eta=1):
        """
        helper function for computing products on (inverse) matrices and vectors
        """

        prods = []
        for i in range(len(vecs)):
            # inverse the matrix 
            mats_inv = np.linalg.inv(mats[i])
            # compute the product
            prods.append(eta*mats_inv@vecs[i]) 

        return prods


if __name__ == "__main__":

    from matplotlib.colors import LinearSegmentedColormap
    # x = ot.Sample([[1], [2], [3], [4], [5]]) # ot.Normal(1).getSample(5)
    # y = ot.Sample([[1], [2], [3], [float('nan')], [5]])

    # print("x=", x)
    # print("y=", y)
    # x, y, n = _remove_nan(x, y)
    # print(x)
    # print(y)
    # print(n)
    # quit()


    # def G(x):
    #     return [-x[0]]
    
    # Define the objective function

        # Define the objective function
    def G(x_p):
        x = x_p[0]-6
        #return [x_p[0]]
        return [0.005*(x**4 +10*x**3+20*x**2+x +2 +160) + 2]  # degree 4 polynomial function

    h = ot.PythonFunction(1, 1, G)

    margin = Normal(6, 1)
    margin.setDescription(ot.Description(['X1']))
    f0 = JointDistribution([margin], ot.IndependentCopula(1))
    f0.setDescription(ot.Description(['X1']))
    # f0 = ExtendedTrNormal(k=0, s=1, lowerBound=-1, upperBound=1)
    # f0 = Normal(m_0, s_0)
    # f0 = TruncatedDistribution(Normal(0, 5), ot.Interval(-1, 1))

    # stepsize
    eta = 0.5

    # number of total iterations (i.e. number of times the distribution is updated)
    nbIter = 20
    colors = LinearSegmentedColormap.from_list("custom", ["dodgerblue", "red"], N=nbIter)

    # sample size from the current distribution at each iteration
    n_k = 5

    # define the optimization problem 
    opt_problem = RiemannianStochasticOptimization(h, f0, ['X1'], ['X1'], f0)

    samp = Normal(6, 1)
    samp.setDescription(['X1'])
    sampling_dist=JointDistribution([samp], ot.IndependentCopula(1))
    sampling_dist.setDescription(ot.Description(['X1']))

    iterations, quantiles = opt_problem.superquantileSGD(sampling_dist, eta=0.005, nbIter=1000, alpha=0.75, gamma=0.75, q0=0)

    iter_params = []
    for f in iterations:
        param = f.marginals[0].getParameter()
        print(param)
        iter_params.append(param)
    
    iter_params = np.array(iter_params)

    plt.plot(iter_params[:,0], iter_params[:,1])
    plt.scatter(iter_params[:,0], iter_params[:,1])
    plt.scatter(iter_params[-1,0], iter_params[-1,1], color="red")
    plt.show()

    plt.figure(figsize=(16,14))

    X = np.linspace(-2,15,300)
    plt.ylim(-0.5,5.5)
    #plt.xlim(-1.5,10.5)
    plt.plot(X, [G([x])[0] for x in X], color='green', lw=3, label = 'objective function $G$')

    plt.plot(X, [iterations[0].computePDF(x) for x in X], color = "dodgerblue", lw=1.8, label=f"initial dist")
    for i, f in enumerate(iterations[1:]):
        #if i%2 ==0:
        plt.plot(X, [f.computePDF(x) for x in X], color=colors(i/nbIter), lw=1)


    # for sample in samples:
    #     plt.scatter(sample, np.zeros(len(sample)),  color=colors[i])
    m_f = iterations[-1].getParameter()[0][0]
    s_f = iterations[-1].getParameter()[0][1]


    plt.plot(X, [iterations[-1].computePDF(x) for x in X], color = "red", lw=1.8, label=f"final dist. $N({m_f:.2},{s_f:.2})$", )

    plt.legend(fontsize=25)
    plt.show()

    plt.plot(np.arange(len(quantiles)), quantiles)
    plt.show()

    quit()

    def G(x_p):
        x = x_p[0]-6
        return [-0.0005*(x**4 +10*x**3+20*x**2+x +2 +160) + 2]  # degree 4 polynomial function

    h = ot.PythonFunction(2, 1, G)

    f = Normal(2, 1)
    f.setDescription(ot.Description(['X1']))

    partialDist = JointDistribution([f], ot.IndependentCopula(1))
    partialDist.setDescription(ot.Description(['X1']))
    fixedDist = TruncatedDistribution(Normal(2, 3), ot.Interval(4, 5))
    fixedDist.setDescription(ot.Description(['X2']))

    fullDist = JointDistribution(
        [partialDist, fixedDist], ot.IndependentCopula(2)
        )
    fullDist.setDescription(ot.Description(['X1', 'X2']))


    test_prob = RiemannianStochasticOptimization(h, fullDist, ['X1', 'X2'], ['X1'], partialDist)

    nbIter = 10
    Alpha = list(0.5*np.ones(int(nbIter/3)+1)) + list(0.75*np.ones(int(nbIter/3)+1)) + list(0.95*np.ones(int(nbIter/3)+1))
    
    def phi(x):
        return [-x[0]**2]

    iterations, X_samples, Y_samples, perturbedInputs, kwargs_inputs = test_prob.expectationBasedOptimization(
        phi=ot.PythonFunction(1, 1, phi), eta=0.1, nbIter=nbIter, n=100, sampleReuse=False, recycleDepth=3
    )

    for f in iterations:
        print(f.getParameter())

    base_dir = "/home/bketema/Python workspace/Cathare simulations"
    
    quit()
    # df_iterations, df_Ysamples, df_perturbed_inputs, df_kwargs_inputs, time_tag = 
    # test_prob.store_optimization_results(
    #     iterations, Y_samples, kwargs_inputs, base_dir
    #     )
    # quit()

    # df_iterations.to_csv(f"/home/bketema/Python workspace/Cathare simulations/optimization_results/iterations_{time_tag}.csv")
    # df_kwargs_inputs.to_csv(f"/home/bketema/Python workspace/Cathare simulations/optimization_results/kwargs_inputs_{time_tag}.csv")
    # print(time_tag)
    # print(df_iterations)
    # print(df_kwargs_inputs)


    def stochasticGradientAscent(
            self, phi: ot.PythonFunction, eta=3, nbIter=100, resetTime: int=10
    ):
        
        partialDistribution = self.partialDistribution
        fullDistribution = self.fullDistribution

        # objective function definition
        h = self.objectiveFunction

        #initialize empty lists for storing the iterations, 
        # input and output samples (later used for emp. quantile and emp. average computation)
        iterations = [] # list for the iterations on the partial distributions
        fullIterations = [] # list for the iterations on the full distributions (important for sample reuse)
        X_samples = []
        Y_samples = []

        iterations.append(partialDistribution)
        fullIterations.append(fullDistribution)

        for k in range(1, nbIter+1):
            for m in range(resetTime):
                pass
                # compute the gradient

                # update the partial distribution with geodesics

                # update the full distribution

    



    # def naturalEvolutionStrategy(
    #         self, startingDist, eta, iterations=1000, populationSize = 10, startingParam = None
    # ):
        
    #     """
    #     Riemannian SGD (natural gradient) on the initially specified parametric family 
    #     eta: the learning rate
    #     iterations: is the number of iterations
    #     """
        
    #     f = startingDist
    #     list_f = []
    #     list_f_averaged = []

    #     list_f.append(f)
    #     list_f_averaged.append(f)

    #     func = self.objectiveFunction

    #     for i in range(iterations):
    #         population = f.getSample(populationSize) # get the population sample
    #         evals = [func(x) for x in population] # evaluate the function on the population
    #         vanillaGradients = [f.computeLogPDFGradient(x) for x in population] # compute the vanilla grad
    #         finv = np.linalg.inv(f.fisherInformation())   # inverse Fisher metric

    #         naturalGradients = [np.dot(finv, v) for v in vanillaGradients] # convert to the Riemannian grad

    #         v = np.matmul(evals, naturalGradients)/populationSize  # compute the tangent vector update direction

    #         # if retraction == False:    
    #         #     f = f.exponentialMap(-eta/i * v) # Riemannian gradient update
    #         #     list_f.append(f)

            
    #         f = f.retraction(-eta/i * v) # Riemannian gradient update with retraction (Natural grad)
    #         list_f.append(f)

    #         # Polyak averaging
    #         f_past = list_f[-2] 
    #         f_averaged = f_past.exponentialMap(1/i * f_past.logarithmicMap(f))
    #         list_f_averaged.append(f_averaged)

    #         # # logging
    #         # if i%10 == 0:
    #         #     print()

    #     return list_f, 

    @staticmethod
    def reconstruct_optimization_results_from_dataframe(df_iterations: pd.DataFrame, df_Ysamples):
        Y_samples = np.array(df_Ysamples)
        df = df_iterations
        nbIter = df_Ysamples.shape[0]

        iterations = []
        for k in range(1, nbIter):
            partial_dist_list = []
            for column in df.columns:
                if df.loc[0, column][0] == "TruncatedDistribution":
                    cls = getattr(ot, "TruncatedDistribution")
                    baseline = getattr(ot, df.loc[0, column][1])
                    param, inter = df.loc[k, column][0:1], df.loc[k, column][2:3]

                    dist = cls(baseline(*param), ot.Interval(*inter))
                    dist.setDescription(ot.Description([column]))
                    partial_dist_list.append(dist)

                else:
                    param = df.loc[k, column]
                    name = df.loc[0, column]
                    cls = getattr(ot, name)

                    dist = cls(*param)
                    dist.setDescription(ot.Description([column]))
                    partial_dist_list.append(dist)

                joint = JointDistribution(partial_dist_list, ot.IndependentCopula(df.shape[1]))
                iterations.append(joint)
                
        return iterations, Y_samples


    def sgdLikelihoodWeights(self, referenceDistribution, iterations):
        
        f0 = referenceDistribution

        for i in range(iterations):
            
            pass

    def superquantileRSGD(
            self, G, startingDist, startingQuantile, eta, alpha, iterations=1000, populationSize = 1
    ):
        f = startingDist
        q = startingQuantile
        

        for k in range(1, iterations):
            # inverse Fisher information
            J_inv = np.linalg.inv(f.fisherInformation())
            
            # sample from f and compute H on the sample
            x_k = f.getSample(1)
            x_k_float = np.array(x_k).transpose()[0][0]
            y_k = G(x_k_float)

            # riemannian gradient for both f and v components
            # gradLogf = J_inv@np.array(f.computeLogPDFGradient(x_k)).transpose()
            gradLogf = J_inv@np.array(f.computeLogPDFGradient(x_k)).transpose()
            partialH = RiemannianStochasticOptimization.partialq_H(y_k, alpha, q)
            
            v_q = eta/k * partialH 
            v_f = eta/k * RiemannianStochasticOptimization.H(y_k, alpha, q)* gradLogf.transpose()[0]
            print("v_q =", v_q)
            print("v_f =", v_f)
            q = q + v_q
            print("q=",q)
            f = f.exponentialMap(v_f)
            
            s = f.getParameter()[1]
            if s < 0.005:
                return f, q
            
            print("k = ", k)
        return f, q
    

    def weightedSuperQuantileRSGD(
            self, startingDist: ot.Distribution, f_nominal, startingQuantile, eta, alpha, X_sample, Y_sample
    ):
        """
        f_nominal: baseline distribution generating the sample
        """
        f0 = startingDist
        q0 = startingQuantile

        Y = Y_sample
        X = X_sample

        iterations = len(Y)

        f = f0
        q = q0
        for k in range(1, iterations):
            # inverse Fisher information at f
            J_inv = np.linalg.inv(f.fisherInformation())
            print("fisher inv=", J_inv)
            x_k = X[k]
            # build the riemannian gradient for both component f and q
            gradf = np.array(f.computePDFGradient(x_k)).transpose()
            gradL = J_inv@gradf/f_nominal.computePDF(x_k)
            print("k = ", k)
            print("gradf=", gradf)
            print("gradL=", gradL)
            partialH = RiemannianStochasticOptimization.partialq_H(Y[k], alpha, q)

            # define the updates using the exponential map for f and the Euclidean step for q
            v_q = eta/k * partialH *f.computePDF(X[k])/f_nominal.computePDF(X[k])
            v_f = eta/k * RiemannianStochasticOptimization.H(Y[k], alpha, q)* gradL.transpose()
            print("v_q =", v_q)

            q = q + v_q
            print("q=",q)
            f = f.exponentialMap(v_f)
            
        return f, q



if __name__ == "__main__":

    Y = np.arange(1, 30, 1)
    y = 20

    L = RiemannianStochasticOptimization.indicator(Y, y)
    print(L)

    quit()

    def G(x):
        return [x[0]**2 + 1]
    
    c = ot.IndependentCopula(2)
    f = Normal(0, 1)
    f.setDescription(ot.Description(["X1"]))
    g = ot.Uniform(-1, 1)
    g.setDescription(ot.Description(["X2"]))

    iterations = [JointDistribution([f, g], c) for s in range(1, 6)]
    Y_samples = [np.arange(1, 12) for s in range(10)]

    test_prob = RiemannianStochasticOptimization(G, iterations[0], ["X1", "X2"], ["X1", "X2"], iterations[0].marginals[0])

    df_it, df_Y, _ = test_prob.store_optimization_results(iterations, Y_samples)
    print("")
    print(df_it)
    print(df_Y)

    quit()

    @staticmethod
    def reconstruct_optimization_results(df):
        """
        This method does the opposite of the store_optimization_results staticmethod
        """
        

