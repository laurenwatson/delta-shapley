import argparse
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from data import *
from models import *
import math
import pandas as pd
import time
from sklearn.metrics import f1_score, roc_auc_score, log_loss


class MonteCarloShapley():
    """
    Take the algorithm from Ghorbani (Data Shapley) and adapted it
    """

    def __init__(self, X_train, y_train, X_test, y_test, L, beta, c, a, b, sup, num_classes, params):
        """
        Args:
            trainset: the whole dataset from which samples are taken
            testset: the validation set
            datapoint: datapoint to be evaluated (index)
            L: Lipschitz constant
            beta: beta-smoothness constant
            c: learning rate at step 1, decaying with c/t
            a: the "a" parameter in the (a,b)-bound in the Shapley value estimation
            b: the "b" parameter in the (a,b)-bound in the Shapley value estimation
            sup: the supremum of the loss function
            num_classes:
            params
        """
        self.X = X_train
        self.X_test = X_test
        self.y = y_train
        self.y_test = y_test
        self.L = L
        self.beta = beta
        self.c = c
        self.shapley = 0
        self.a = a
        self.b = b
        self.sup = sup
        self.n = params.datasize
        self.De = params.datasize
        self.num_classes = num_classes
        self.params = params
        self.SVs = []
        self.samples = []
        self.SVdf = None

    def includes_all_classes(self, num_classes, indices):
        ys=self.y[indices]
        # print('checking new permutation', num_classes, ys, type(ys[0]))
        for i in np.arange(num_classes):
            if i not in ys:
                return False
        return True

    def run(self, datapoints, params):
        """
        Args:
            datapoint: the index of the datapoint in the trainset to be evaluated
            return: the approximate Shapley value
        """
        self.SVdf = pd.DataFrame(columns = sum([[str(i) + "_SV",str(i) + "_time",str(i) + "_layer"] for i in datapoints],[]))
        shapley_values = np.zeros(len(datapoints))
        iter = 1
        while (not self.check_convergence_rolling(iter, datapoints)) and iter < 2 * 10e5:
            row_iteration = dict()
            if iter % 10 == 0:
                print("Monte Carlo running in iteration {}".format(iter))
            for i in range(len(datapoints)):
                datapoint = datapoints[i]
                if len(self.SVdf) > 0:
                    est_shapley = self.SVdf.iloc[-1][str(datapoint)+"_SV"]
                else:
                    est_shapley = 0

                permutation = np.arange(self.n)
                np.random.shuffle(permutation)
                # see https://www.geeksforgeeks.org/how-to-find-the-index-of-value-in-numpy-array/
                datapoint_index = np.where(permutation == datapoint)[0][0]

                # prevent the evaluated datapoint from being the first in the permutation
                while datapoint_index == 0:
                    np.random.shuffle(permutation)
                    # see https://www.geeksforgeeks.org/how-to-find-the-index-of-value-in-numpy-array/
                    datapoint_index = np.where(permutation == datapoint)[0][0]

                indices = permutation[:datapoint_index]

                while not self.includes_all_classes(self.num_classes, indices):
                    permutation = np.arange(self.n)
                    np.random.shuffle(permutation)
                    # see https://www.geeksforgeeks.org/how-to-find-the-index-of-value-in-numpy-array/
                    datapoint_index = np.where(permutation == datapoint)[0][0]

                    # prevent the evaluated datapoint from being the first in the permutation
                    while datapoint_index == 0:
                        np.random.shuffle(permutation)
                        # see https://www.geeksforgeeks.org/how-to-find-the-index-of-value-in-numpy-array/
                        datapoint_index = np.where(permutation == datapoint)[0][0]

                    indices = permutation[:datapoint_index]

                self.samples = datapoint_index
                time_now = time.time()
                v = self.compute_MC(indices, datapoint, self.num_classes, self.params)
                elapsed_time = time.time() - time_now
                est_shapley = est_shapley * ((iter - 1)/iter) + (v/iter)
                shapley_values[i] = est_shapley
                row_iteration[str(datapoint) + "_SV"] = est_shapley
                row_iteration[str(datapoint) + "_time"] = elapsed_time
                row_iteration[str(datapoint) + "_layer"] = datapoint_index
            row_iteration_df = pd.DataFrame([row_iteration])
            self.SVdf = pd.concat([self.SVdf, row_iteration_df], ignore_index=True)
            iter +=1
            if os.path.exists(params.save_dir + "/MonteCarlo_FM_CNN.csv"):
                os.remove(params.save_dir + "/MonteCarlo_FM_CNN.csv")
        self.SVdf.to_csv(params.save_dir+"/MonteCarloSampling.csv")
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
        # indices_incl_datapoint[random_idx] = datapoint_idx
        # train model
        sample_datapoint_X = self.X[indices_incl_datapoint]
        sample_datapoint_y = self.y[indices_incl_datapoint]
        model_datapoint = return_model(params, num_classes)
        trained_datapoint = self.train(model_datapoint, sample_datapoint_X,sample_datapoint_y)

        marginal_contribution = self.evaluate(trained, trained_datapoint, params)
        return marginal_contribution

    def train(self, model, X, y):
        """
        Training loop for NNs
        Args:
            model:
            optimizer:
            dataloader:
            params:
        Return model
        """
        model.fit(X, y)

        return model

    def check_convergence(self, iteration, estimated_shapley, estimated_SVs):
        if iteration < 100:
            return False
        elif iteration >5000:
            return True
        else:
            estimated_shapley_old = estimated_SVs[iteration-100]
            if estimated_shapley_old != 0:
                rel_diff = (abs(estimated_shapley-estimated_shapley_old)/abs(estimated_shapley))
                if iteration % 100 == 0:
                    print("Iteration: {}, relative difference: {}".format(iteration, rel_diff))
                return rel_diff < 0.05
            else:
                return False

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
        return validation_loss - validation_loss_datapoint

    def evaluate_model(self, model,  X,y):
        """
        Computes the validation loss of a model
        """

        return log_loss(y, model.predict_proba(X))


    def check_convergence_rolling(self, iteration, datapoints):
        if iteration < 101:
            return False
        else:
            current_row = self.SVdf.iloc[-1]
            old_row = self.SVdf.iloc[-100]
            small_deviation = [self.check_deviation((old_row[str(i)+"_SV"], current_row[str(i)+"_SV"])) for i in datapoints]
            if iteration % 100 == 0:
                print("Iteration {}, current convergence: {}/10".format(iteration, sum(small_deviation)))
            if (sum(small_deviation)/(len(datapoints))) < 0.05:
                return True
            else:
                return False

    def check_deviation(self, vals):
        old = vals[0]
        new = vals[1]
        # check whether either is 0 to avoid div by 0, return False if either is 0
        if old == 0 or new == 0:
            return 1e6
        ratio = abs(new - old) /abs(new)
        return ratio
