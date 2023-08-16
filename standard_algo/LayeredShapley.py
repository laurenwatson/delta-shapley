from torch.utils.data import Subset, DataLoader
import argparse
from data import *
from models import *
import math
import torch
import torch.nn.functional as F
from torch import optim


class LayeredShapley():

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
        self.n = 100
        self.De = len(testset)
        self.num_classes = num_classes
        self.params = params
        self.f =  open("../mk.csv", "w")
        self.f.write("layer,samples\n")

    def run(self, datapoint):
        """
        Args:
            datapoint: the index of the datapoint in the trainset to be evaluated
            return: the approximate Shapley value
        """
        varphi = 0
        total_m = 0
        for i in range(1, self.n):
            m = self.compute_mk(i, i) # assuming T = i?
            self.log_mk(i, m)
            total_m += m
            w = math.factorial(self.n - 1)/((math.factorial(i)*math.factorial(self.n - 1 - i)))
            sum_v = 0
            # use torch subset to sample a subset from data using indices
            for j in range(math.ceil(m)):
                print("Layer: {}, mk: {}, current coalition: {}".format(i, m, j))
                indices = np.random.choice(a = self.n, size = i)
                v = self.compute_MC(indices, datapoint, self.num_classes, self.params)
                sum_v += v
            varphi += (1/m)*sum_v
            print("Varphi in this layer is {}".format(varphi))

        estimated_shapley = varphi / self.n
        #print("total number of coalitions sampled: " + str(total_m))
        return estimated_shapley


    def compute_MC(self, indices, datapoint_idx, num_classes, params):
        """
        Compute the marginal contribution of a datapoint to a sample
        Args:
            indices: the indices of the train dataset to be used as a sample
            datapoint: the index of the evaluated datapoint
        """
        indices_incl_datapoint = np.append(indices,datapoint_idx)
        np.random.shuffle(indices_incl_datapoint)

        # extract the sample from the dataset
        sample = Subset(self.trainset, list(indices))
        sample_datapoint = Subset(self.trainset, list(indices_incl_datapoint))
        trainloader = DataLoader(sample, batch_size=1, shuffle=True, num_workers=2)
        trainloader_datapoint = DataLoader(sample_datapoint, batch_size=1, shuffle=True, num_workers=2)

        model = return_model(params, num_classes)
        model_datapoint = return_model(params, num_classes)

        trained = self.train(model, trainloader, params)
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
                output = None
                loss = None
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
        testloader = DataLoader(self.testset, batch_size=1, shuffle=False, num_workers=2)
        validation_loss = self.evaluate_model(model, testloader, params)
        validation_loss_datapoint = self.evaluate_model(model_datapoint, testloader, params)
        return validation_loss_datapoint - validation_loss

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

        return overall_test_loss

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
        return 2*math.log((2*self.n)/self.b)*(((H_s**2/self.De) + 2*self.sup*self.a/3)/self.a**2)

    def log_mk(self,i,m):
        self.f.write(str(i) + "," + str(m) + "\n")
