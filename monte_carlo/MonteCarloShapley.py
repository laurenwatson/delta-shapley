from torch.utils.data import Subset, DataLoader, SequentialSampler
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
import torch
import torch.nn.functional as F
from torch import optim
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR
import time
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MonteCarloShapley():
    """
    Take the algorithm from Ghorbani (Data Shapley) and adapted it
    """

    def __init__(self, trainset, testset, L, beta, c, a, b, sup, num_classes, params):
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
        self.trainset = trainset
        self.testset = testset
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
            if os.path.exists(params.save_dir + "/MonteCarlo_FM_CNN_n100_2.csv"):
                os.remove(params.save_dir + "/MonteCarlo_FM_CNN_n100_2.csv")
            self.SVdf.to_csv(params.save_dir+"/MonteCarlo_FM_CNN_n100_2.csv")
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
        sample = Subset(self.trainset, list(indices))
        sampler = SequentialSampler(sample)
        trainloader = DataLoader(sample, batch_size=1, sampler=sampler)
        model = return_model(params, num_classes)
        trained = self.train(model, trainloader, params)

        # insert in differential datapoint
        if random_idx == 0:
            indices_incl_datapoint = np.concatenate(([datapoint_idx], indices))
        else:
            indices_incl_datapoint = np.concatenate((np.concatenate((indices[:random_idx], [datapoint_idx])), indices[random_idx:]))
        # train model
        sample_datapoint = Subset(self.trainset, list(indices_incl_datapoint))
        sampler_datapoint = SequentialSampler(sample_datapoint)
        trainloader_datapoint = DataLoader(sample_datapoint, batch_size=1, sampler=sampler_datapoint)
        model_datapoint = return_model(params, num_classes)
        trained_datapoint = self.train(model_datapoint, trainloader_datapoint, params)
        marginal_contribution = self.evaluate(trained, trained_datapoint, params)
        return marginal_contribution

    def train(self, model, dataloader, params):
        """
        Training loop for NNs
        Args:
            model:
            optimizer:
            dataloader:
            params:
        Return model
        """
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0)
        lr_lambda = lambda epoch: params.learning_rate / (epoch + 1)
        #scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        for epoch in np.arange(params.epochs):
            for (images, targets) in dataloader:
                model.zero_grad()
                optimizer.zero_grad()
                if params.device != 'cpu':
                    images = images.cuda()
                    targets = targets.cuda()
                output = model(images)
                loss = F.cross_entropy(output, targets, reduction='mean')
                loss.backward()
                optimizer.step()
                #scheduler.step()
                output = None
                loss = None
        return model

    def check_convergence(self, iteration, estimated_shapley, estimated_SVs):
        if iteration < 100:
            return False
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
        testloader = DataLoader(self.testset, batch_size=params.batch_size, shuffle=False)
        validation_loss = self.evaluate_model(model, testloader, params)
        validation_loss_datapoint = self.evaluate_model(model_datapoint, testloader, params)
        return validation_loss - validation_loss_datapoint

    def evaluate_model(self, model, testloader,params):
        """
        Computes the validation loss of a model
        """
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0)
        with torch.no_grad():
            losses = []
            for (images, targets) in testloader:
                model.zero_grad()
                optimizer.zero_grad()
                if params.device != 'cpu':
                    images = images.cuda()
                    targets = targets.cuda()
                output = model(images)
                loss = F.cross_entropy(output, targets, reduction='sum')
                losses.append(loss.clone().detach())
                output = None
                loss = None
            overall_test_loss = sum(losses) / self.testset.__len__()

        return overall_test_loss.item()

    def check_convergence_rolling(self, iteration, datapoints):
        if iteration < 102:
            return False
        else:
            current_row = self.SVdf.iloc[-1]
            old_row = self.SVdf.iloc[-101]
            small_deviation = [self.check_deviation((old_row[str(i)+"_SV"], current_row[str(i)+"_SV"])) for i in datapoints]
            if iteration % 100 == 0:
                print("Iteration {}, current convergence: {}".format(iteration, sum(small_deviation)/len(datapoints)))
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