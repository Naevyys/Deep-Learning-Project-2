import torch


class SGD():

    def __init__(self, lr=1e-2, batch_size=None):

        self.lr = lr 
        self.batch_size = batch_size 


    def step(self, list_params): 
        """
        Run the stochastic gradient descent and update the parameters of the model
        : list_params: List of list of tuples of tensors (weights, gradient)
            First list contains the layers, the second list contains tuples (weights, gradient)
            of the different parameters of the given layer
        :return updated_params: A list containing the update parameters of the network
        """

        assert self.batch_size is not None, "You forgot to assign a batch size in the training function!"

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
                    param = param - self.lr*gradient/self.batch_size
                    intermediate_param.append(param)
                updated_params.append(intermediate_param)
            else:
                # If no parameters, just return an empty list
                updated_params.append([])

        return updated_params