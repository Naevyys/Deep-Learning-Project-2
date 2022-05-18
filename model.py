import torch  # Only to disable autograd
from torch import empty
import time
import datetime
import os
import pathlib

from .src.Layers.convolution import Conv2d
from .src.Layers.relu import ReLU
from .src.Layers.sigmoid import Sigmoid 
from .src.sequential import Sequential
from .src.Loss_functions.mse import MSELoss
from .src.utils import waiting_bar

torch.set_grad_enabled(False)

class Model():
    def __init__(self):
        """
        Instantiates the model class.
        :return: None
        """
        
        self.Conv2d = Conv2d
        self.ReLU = ReLU
        self.Sigmoid = Sigmoid
        # TODO add Upsampling
        self.Sequential = Sequential(Conv2d(3,3,3,1,0,0,True), ReLU(), Conv2d(3,3,3,1,0,0,True), Sigmoid())

        self.lr = 1e-3 
        self.batch_size = None
        self.eval_step = 5
        self.best_model = None
        self.path = str(pathlib.Path(__file__).parent.resolve())
        # To store the training logs 
        # First row: the epoch number
        # Second row: the training error 
        # Third row: the validation error
        self.logs = [[], [], []]
       

    def load_pretrained_model(self):
        """
        Loads best model from file bestmodel.pth
        :return: None
        """
        # The path needed when used in testing mode 
        self.best_model = self.Sequential
        params = torch.load(torch.load(self.path+"/bestmodel.pth", map_location = lambda storage, loc: storage))
        self.best_model.update_param(params)

    def train(self, train_input, train_target, num_epochs=20, batch_size=64, validation=0.2):
        """
        Trains the model.
        :param train_input: Training data.
        :param train_target: Train targets.
        :return: None
        """
        # Update the batch size 
        self.batch_size = batch_size
        # Custom train/validation split - Start by shuffling and sending to GPU is available 
        idx = torch.randperm(train_input.size()[0])
        train_input = train_input[idx, :, :, :]
        train_target = train_target[idx, :, :, :]
        # Then take the last images as validation set (w.r.t. proportion)
        split = int(validation * train_input.size(0))
        # Training data is standardized by the DataLoader 
        val_input = (train_input[0:split] / 255)
        val_target = (train_target[0:split] / 255)
        train_input = (train_input[split:-1]) / 255
        train_target = (train_target[split:-1] / 255)
    
        nb_images_train = len(train_input)
        nb_images_val = len(val_input)

        # Monitor time taken
        start = time.time()
        # The loop on the epochs
        for epoch in range(0, num_epochs):
            idx = torch.randperm(nb_images_train)
            # Shuffle the dataset at each epoch TODO check if faster to call data_iter for each batch
            for train_img, target_img in zip(torch.split(train_input, batch_size),
                                             torch.split(train_target, batch_size)):
                # TODO implement the equivalent
                #loss = criterion(output, target_img)
                #self.model.zero_grad()
                #loss.backward()
                #optimizer.step()
                output = self.Sequential(train_img)
                # TODO Do we really need the loss here?
                loss = MSELoss.forward(output, target_img)
                d_loss = MSELoss.backward(output, target_img)
                # Compute the gradient
                self.Sequential.backward(d_loss)
                # Compute the SGD and update the parameters
                self.optimise(self.Sequential.param())


            # Evaluate the model every eval_step
            if (epoch + 1) % self.eval_step == 0:
                with torch.no_grad():
                    eva_batch_size = 1000
                    train_error = 0.
                    val_error = 0.
                    # Computing the number of split to compute the mean of the error of each batch
                    if nb_images_train%eva_batch_size == 0:
                        nb_split_train = nb_images_train//eva_batch_size
                    else:
                        nb_split_train = nb_images_train // eva_batch_size + 1

                    if nb_images_val%eva_batch_size == 0:
                        nb_split_val = nb_images_val//eva_batch_size
                    else:
                        nb_split_val = nb_images_val // eva_batch_size + 1

                    train_zip = zip(torch.split(train_input, eva_batch_size),
                                    torch.split(train_target, eva_batch_size))
                    val_zip = zip(torch.split(val_input, eva_batch_size), torch.split(val_target, eva_batch_size))

                    for train_img, target_img in train_zip:
                        train_error += MSELoss.forward(self.Sequential(train_img), target_img)

                    for val_img, val_img_target in val_zip:
                        val_error +=MSELoss.forward(self.Sequential(val_img), val_img_target)

                    train_error = train_error / nb_split_train
                    val_error = val_error / nb_split_val

                    self.logs[0].append(epoch)
                    self.logs[1].append(train_error)
                    self.logs[2].append(val_error)

                waiting_bar(epoch, num_epochs, (self.logs[1][-1], self.logs[2][-1]))

        # Save the model - path name contains the parameters + date
        date = datetime.now().strftime("%d%m%Y_%H%M%S")
        path = str(self.lr) + "_" + str(self.batch_size) + "_" + date + ".pth"

        torch.save(self.Sequential.param(), self.path +"/outputs/trained_models/"+ path)
        # Save the logs as well
        self.logs = torch.tensor(self.logs)
        torch.save(self.logs, self.path + "/outputs/logs/" + path)

        # Record and print time
        end = time.time()
        min = (end - start) // 60
        sec = (end - start) % 60
        print("\nTime taken for training: {:.0f} min {:.0f} s".format(min, sec))
        del train_input, train_target, train_input, train_target

    def predict(self, test_input):
        """
        Predicts with the model on the provided input.
        :param test_input: Test input.
        :return: The prediction (torch.Tensor).
        """
        out = self.Sequential(test_input.float()/255.0)
        # Rescale the output between 0 and 255 - need to be optimised though 
        min = out.min()
        max = out.max()-min
        return ((out - min ) / (max))*255

    def optimise(self, list_params): 
        """
        Run the stochastic gradient descent and update the parameters of the model
        : list_params: List of list of tuples of tensors (weights, gradient)
            First list contains the layers, the second list contains tuples (weights, gradient)
            of the different parameters of the given layer
        :return: None
        """
        updated_params = []
        # Iterate on the layer's parameters  
        for layer_param in list_params:
            # Check whether the list is empty 
            if layer_param:
                intermediate_param = []
                # Iterate on the parameters of the layer
                for param, gradient in layer_param:
                    # Use the update rule from the stochastic descend gradient
                    # Thre gradient already contains the sum of all input from the batch
                    # We need to normalise (or get the average) by dividing by the batch size
                    param -= self.lr*gradient/self.batch_size
                    intermediate_param.append(param)
                updated_params.append(intermediate_param)
            else:
                # If no parameters, just return an empty list
                updated_params.append([])
        # Assign the newly calculated parameters
        self.Sequential.update_param(updated_params)

    def psnr(self, denoised, ground_truth):
        """
        Computes the Peak Signal-to-Noise Ratio of a denoised image compared to the ground truth.
        :param denoised: Denoised image. Must be in range [0, 1].
        :param ground_truth: Ground truth image. Must be in range [0, 1].
        :return: PSNR (0-dimensional torch.Tensor)
        """

        assert denoised.shape == ground_truth.shape, "Denoised image and ground truth must have the same shape!"

        mse = torch.mean((denoised - ground_truth) ** 2)
        return -10 * torch.log10(mse + 10 ** -8)
    
    