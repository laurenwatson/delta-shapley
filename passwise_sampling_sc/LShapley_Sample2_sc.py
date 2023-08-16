import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from data_utils import *
from models import *
import math
import pandas as pd
import numpy as np
import time
from sklearn.metrics import f1_score, roc_auc_score, log_loss

class LayeredShapley_Sample2():
    """
    A refined sampling procedure. In each iteration of the algorithm, coalitions from each layer are sampled, where
    the layer with the lowest m_k has one coalition used, with all other layers have coalitions sampled in proportion.
    The algorithm terminates on convergence or sampling of all coalitions.
    """

    def __init__(self, num_classes, params):
        """
        Args:
            trainset: the whole dataset from which samples are taken
            testset: the validation set
            datapoint: datapoint to be evaluated (index)
            L: Lipschitz constant
            beta: beta-smoothness constant
            c: learning rate at step 1, decaying with ~c/t
            sup: the supremum of the loss function
            num_classes:
            params
        """

        self.n = params.datasize
        self.De = params.datasize
        self.num_classes = num_classes
        self.params = params
        self.f =  open("layers.csv", "w")
        self.f.write("layer,value\n")
        self.SVs = []
        self.random_loss = self.calculate_random_loss(num_classes)
        self.SVdf = pd.DataFrame()
        X,y=load_data(params.dataset)
        train_size=int(len(X)*0.8)
        self.X=X[:train_size]
        self.y=y[:train_size]
        self.X_test=X[train_size:]
        self.y_test=y[train_size:]

    def doesnt_include_all_classes(self, num_classes, indices):
        ys=self.y[indices]
        # print('checking new permutation', num_classes, ys, type(ys[0]))
        for i in np.arange(num_classes):
            if i not in ys:
                return True
        return False

    def calculate_random_loss(self, num_classes):
        return log_loss([1,0], [[0.5,0.5],[0.5,0.5]])

    def run(self, eval_datapoint_ids, params):
        """
        Args:
            amount_datapoints: the number of datapoints to be evaluated
            return: the approximate Shapley value
        """

        # draw
        datapoints = eval_datapoint_ids
        p = self.calculate_layer_probabilities(self.n)
        shapley_values = np.zeros(self.n)
        for i in range(len(eval_datapoint_ids)):
            datapoint = eval_datapoint_ids[i]
            varphi = 0
            # shapley value from 100 iterations ago
            iter = 1
            est_shapley = 0
            estimated_SVs = []
            layers = []
            times = []
            while (not self.check_convergence_rolling(iter, estimated_SVs)) and iter < 2*10e5:
                sum_v = 0
                # sample a layer
                layer = np.random.choice(np.arange(2,self.n),p = p)
                # sample a coalition:
                sample = np.random.choice(self.n, size = layer, replace=False)

                # see if the evaluated datapoint is in the sample
                while (sum(np.isin(sample, datapoint)) > 0) or self.doesnt_include_all_classes(self.num_classes, sample):
                    # sample again
                    sample = np.random.choice(self.n, size=layer, replace=False)

                time_now = time.time()
                # compute marginal contribution
                v = self.compute_MC(sample, datapoint, self.num_classes, self.params)
                elapsed_time = time.time()-time_now
                # compute new estimated shapley value
                est_shapley = ((iter-1)*est_shapley/iter) + (v/iter)
                self.log_layer(layer, est_shapley)
                print("Datapoint {} Estimated Shapley in iteration {} is {}".format(datapoint, layer, est_shapley), flush=True)
                iter += 1
                estimated_SVs.append(est_shapley)
                layers.append(layer)
                times.append(elapsed_time)
                shapley_values[datapoint] = est_shapley
            self.SVs.append(estimated_SVs)
            self.SVdf[str(datapoint) + "_SV"] = pd.Series([estimated_SVs][0])
            self.SVdf[str(datapoint) + "_time"] = pd.Series([times][0])
            self.SVdf[str(datapoint) + "_layer"] = pd.Series([layers][0])
        #SV_df = pd.DataFrame(data = self.SVs, columns = [str(i) for i in range(len(datapoints))])
        self.SVdf.to_csv(params.save_dir + "/Shapley_Values_Sampling2.csv")
        self.f.close()
        return shapley_values


    def compute_MC(self, indices, datapoint_idx, num_classes, params):
        """
        Compute the marginal contribution of a datapoint to a sample
        Args:
            indices: the indices of the train dataset to be used as a sample
            datapoint: the index of the evaluated datapoint
        """
        # compute a random point to insert the differing datapoint
        random_idx = np.random.randint(0, len(indices))

        # train model without datapoint
        sample_X = self.X[indices]
        sample_y = self.y[indices]
        model = return_model(params, num_classes)
        trained = self.train(model, sample_X,sample_y)

        # swap in differential datapoint
        indices_incl_datapoint = np.append(indices, datapoint_idx)
        # indices_incl_datapoint = indices
        # indices_incl_datapoint[random_idx] = datapoint_idx
        # train model
        sample_datapoint_X = self.X[indices_incl_datapoint]
        Sampler_datapoint_y = self.y[indices_incl_datapoint]
        model_datapoint = return_model(params, num_classes)
        trained_datapoint = self.train(model_datapoint, sample_datapoint_X,Sampler_datapoint_y)

        marginal_contribution = self.evaluate(trained, trained_datapoint, params)
        return marginal_contribution

    def train(self, model, X, y):
        """
        Training loop
        Return model
        """
        model.fit(X, y)

        return model

    def evaluate(self, model, model_datapoint, params):
        """
        Compute the difference in validation loss
        Args:
            testset: evaluation set from which we use the loss to compute
            fullmodel:  the model trained including our evaluated datapoint
            removalmodel: the model trained without our evaluated datapoint
            return: the marginal contribution of the datapoint
        """
        validation_loss = self.evaluate_model(model, self.X_test,self.y_test)
        validation_loss_datapoint = self.evaluate_model(model_datapoint, self.X_test,self.y_test)
        print(validation_loss, self.random_loss)
        # if validation_loss<self.random_loss and validation_loss_datapoint<self.random_loss:
        return validation_loss - validation_loss_datapoint
        # else:
        #     print('Random loss found')
        #     return 0.0

    def evaluate_model(self, model, X,y):
        """
        Computes the validation loss of a model
        """

        return log_loss(y, model.predict_proba(X))

    def compute_mk(self, layersize, T):
        """
        Args:
            layersize: the size of the samples in the considered layer
            T: the number of steps taken
        """
        # using the notation from the paper, compute some terms for better readability
        q = self.beta * self.c
        H_s_1 = self.sup**(q/(q+1))
        H_s_2 = (2*self.c*(self.L**2))**(1/(q+1))
        H_s_3 = T**(q/(q+1))
        H_s_4 = (1+(1/q))/(max(layersize-1, 1))
        H_s = H_s_1*H_s_2*H_s_3*H_s_4
        return math.ceil(2*math.log((2*self.n)/self.b)*(((H_s**2/self.De) + 2*self.sup*self.a/3)/self.a**2))

    def log_layer(self,l,value):
        """
        Function to log layer to check probability distribution of log layers is correct
        """
        self.f.write(str(l) +","+str(value)+"\n")

    def check_convergence(self, iteration, estimated_shapley, estimated_SVs):
        if iteration < 100:
            return False
        else:
            estimated_shapley_old = estimated_SVs[iteration-100]
            if estimated_shapley_old != 0:
                rel_diff = (abs(estimated_shapley_old-estimated_shapley)/abs(estimated_shapley))
                print("Iteration: {}, relative difference: {}".format(iteration, rel_diff))
                return rel_diff < 0.05
            else:
                return False

    def calculate_layer_probabilities(self, n):
        if self.params.prob_type=='select-layers-middle-third':
            p=[0 for i in range(2,n)]
            layer_floor=math.floor((n-1)/3)
            layer_ceil=math.floor(2*(n-1)/3)
            prob_per_layer=1.0/(layer_ceil-layer_floor)
            for l in np.arange(layer_floor,layer_ceil ):
                p[l]=prob_per_layer
        elif self.params.prob_type=='select-layers-bottom-middle-quarter':
            p=[0 for i in range(2,n)]
            layer_floor=math.floor((n-1)/4)
            layer_ceil=math.floor((n-1)/2)
            prob_per_layer=1.0/(layer_ceil-layer_floor)
            for l in np.arange(layer_floor,layer_ceil ):
                p[l]=prob_per_layer
        elif self.params.prob_type=='select-layers-bottom-middle-tenth':
            p=[0 for i in range(2,n)]
            layer_floor=math.floor(2*(n-1)/10)
            layer_ceil=math.floor(3*(n-1)/10)
            prob_per_layer=1.0/(layer_ceil-layer_floor)
            for l in np.arange(layer_floor,layer_ceil ):
                p[l]=prob_per_layer
        elif self.params.prob_type=='k':
            calculate_C = sum([1/(i-1) for i in range(2,n)])
            p = [(1/(i-1))/(calculate_C) for i in range(2,n)]
        else:
            calculate_C = sum([1/(i-1)**2 for i in range(2,n)])
            p = [(1/(i-1)**2)/(calculate_C) for i in range(2,n)]
        return p

    def check_convergence_rolling(self, iteration, estimated_SVs):
        if iteration < 111:
            return False
        else:
            small_deviation = [self.check_deviation(estimated_SVs[-110 + i], estimated_SVs[-10 + i]) for i in range(10)]
            if iteration:
                print("Iteration {}, current convergence: {}/10".format(iteration, sum(small_deviation)))
            if sum(small_deviation) == 10:
                return True
            else:
                return False

    def check_deviation(self, old, new):
        # check whether either is 0 to avoid div by 0, return False if either is 0
        if old == 0 or new == 0:
            return False
        ratio = abs(new - old) / abs(new)
        if ratio < 0.05:
            return True
        else:
            return False
