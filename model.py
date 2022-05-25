import torch  # Only to disable autograd
from datetime import datetime
import time as time
import pathlib

from .others.src.sgd import SGD
from .others.src.Layers.convolution import Conv2d
from .others.src.Layers.upsampling import Upsampling
from .others.src.Layers.relu import ReLU
from .others.src.Layers.sigmoid import Sigmoid 
from .others.src.sequential import Sequential
from .others.src.Loss_functions.mse import MSELoss
from .others.src.utils import waiting_bar

torch.set_grad_enabled(False)

class Model():
    def __init__(self):
        """
        Instantiates the model class.
        :return: None
        """
        
        # It avoids precision problems, as well as conversion 
        torch.set_default_dtype(torch.float64)
        
        self.SGD = SGD(lr=1e-2, batch_size=32)
        self.Conv2d = Conv2d
        self.ReLU = ReLU
        self.Sigmoid = Sigmoid
        self.Upsampling = Upsampling
        #self.Sequential = Sequential(Conv2d(3,3,3,1,1,1,True), ReLU(), Conv2d(3,3,3,1,1,1,True), Sigmoid())
        self.Sequential = Sequential(
            Conv2d(in_channels=3, out_channels=6, stride=2, padding=1, dilation=1, kernel_size=3), ReLU(),
            Conv2d(in_channels=6, out_channels=9, stride=2, padding=1, dilation=1, kernel_size=3), ReLU(),
            Upsampling(scale_factor=2, in_channels=9, out_channels=6, kernel_size=3, transposeconvargs=False), ReLU(),
            Upsampling(scale_factor=2, in_channels=6, out_channels=3, kernel_size=3, transposeconvargs=False) , Sigmoid())
        self.criterion = MSELoss()

        self.eval_step = 1
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
        params_gradient = torch.load(self.path+"/bestmodel.pth")
        best_params = []
        # params contain both the parameters and gradient, only extract the parameters of the best model
        # Iterate on the layer's parameters  
        for layer_param in params_gradient:
            # Check whether the list is empty 
            if layer_param:
                intermediate_param = []
                # Iterate on the parameters of the layer
                for param, gradient in layer_param:
                    intermediate_param.append(param)
                best_params.append(intermediate_param)
            else:
                # If no parameters, just return an empty list
                best_params.append([])
        # Assign the best parameters
        self.Sequential.update_param(best_params)


    def train(self, train_input, train_target, num_epochs=20, batch_size=32, validation=0.2):
        """
        Trains the model.
        :param train_input: Training data.
        :param train_target: Train targets.
        :return: None
        """
        # Update the batch size 
        self.SGD.batch_size = batch_size
        # Custom train/validation split - Start by shuffling
        idx = torch.randperm(train_input.size()[0])
        train_input = train_input[idx, :, :, :].to(torch.float64)
        train_target = train_target[idx, :, :, :].to(torch.float64)
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
            self.SGD.lr = 10**(2-epoch)
            idx = torch.randperm(nb_images_train)
            for train_img, target_img in zip(torch.split(train_input[idx], batch_size),
                                             torch.split(train_target[idx], batch_size)):
                # Compute the predictions from the model
                output = self.Sequential(train_img)
                # Compute the loss from the predictions
                loss = self.criterion.forward(output, target_img)
                loss_grad = self.criterion.backward()
                # Zero the gradient 
                self.Sequential.zero_grad() 
                # Compute the gradient
                self.Sequential.backward(loss_grad)
                # Compute the SGD and update the parameters
                updated_params = self.SGD.step(self.Sequential.param())
                # Assign the newly calculated parameters
                self.Sequential.update_param(updated_params)


            # Evaluate the model every eval_step
            if (epoch + 1) % self.eval_step == 0:
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
                    train_error += self.criterion.forward(self.Sequential(train_img), target_img)

                for val_img, val_img_target in val_zip:
                    val_error +=self.criterion.forward(self.Sequential(val_img), val_img_target)

                train_error = train_error / nb_split_train
                val_error = val_error / nb_split_val

                self.logs[0].append(epoch+1)
                self.logs[1].append(train_error)
                self.logs[2].append(val_error)

                waiting_bar(i=epoch+1, length=num_epochs, loss=(self.logs[1][-1], self.logs[2][-1]))

        # Save the model - path name contains the parameters + date
        date = datetime.now().strftime("%d%m%Y_%H%M%S")
        path = str(self.SGD.lr) + "_" + str(self.SGD.batch_size) + "_" + date + ".pth"

        torch.save(self.Sequential.param(), self.path +"/others/outputs/trained_models/"+ path)
        # Save the logs as well
        self.logs = torch.tensor(self.logs)
        torch.save(self.logs, self.path + "/others/outputs/logs/" + path)

        # Record and print time
        end = time.time()
        min = (end - start) // 60
        sec = (end - start) % 60
        print("\nTime taken for training: {:.0f} min {:.0f} s".format(min, sec))
        del train_input, train_target

    def predict(self, test_input):
        """
        Predicts with the model on the provided input.
        :param test_input: Test input.
        :return: The prediction (torch.Tensor).
        """
        out = self.Sequential(test_input.double()/255.0)
        # Rescale the output between 0 and 255  
        return out.double()*255

    def psnr(self, denoised, ground_truth):
        """
        Computes the Peak Signal-to-Noise Ratio of a denoised image compared to the ground truth.
        :param denoised: Denoised image. Must be in range [0, 1].
        :param ground_truth: Ground truth image. Must be in range [0, 1].
        :return: PSNR (0-dimensional torch.Tensor)
        """

        assert denoised.shape == ground_truth.shape, "Denoised image and ground truth must have the same shape!"

        assert denoised.shape == ground_truth.shape, "Denoised image and ground truth must have the same shape!"
        return - 10 * torch.log10(((denoised-ground_truth) ** 2).mean((1, 2, 3))).mean()
    
    