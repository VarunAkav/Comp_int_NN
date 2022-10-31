import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os
from functools import reduce
import numpy as np
from copy import deepcopy

# All the models from the Models folder will be stored in a dictionary
# model.layers[7]._inbound_nodes[0].inbound_layers for getting the predecessor layers
modelPaths = dict()
for filepath in os.listdir('Models'):
    filename, ext = os.path.splitext(filepath)
    if ext == '.h5':
        modelPaths[filename] = os.path.join('Models', filepath)


class ModelSummary:

    def __init__(self) -> None:
        self.class_name = ''
        self.name = ''
        self.layers = []
        self.neurons = 0
        self.additions = 0
        self.multiplications = 0
        self.comparisions = 0
        self.connections = 0

    def __repr__(self):
        return json.dumps({
            "class_name": self.class_name,
            "name": self.name,
            "layers": [layer.__repr__() for layer in self.layers],
            "neurons": self.neurons,
            "additions": self.additions,
            "multiplications": self.multiplications,
            "comparasions": self.comparisions,
            "connections": self.connections,
        }, indent=4, separators=(',', ':'))


class LayerSummary:
    def __init__(self) -> None:
        self.class_name = ''
        self.name = ''
        self.shape = []
        self.neurons = 0
        self.additions = 0
        self.multiplications = 0
        self.comparisions = 0
        self.connections = 0

    def __repr__(self):
        return json.dumps({
            "class_name": self.class_name,
            "name": self.name,
            "shape": self.shape,
            "neurons": self.neurons,
            "additions": self.additions,
            "multiplications": self.multiplications,
            "comparasions": self.comparisions,
            "connections": self.connections,
        }, indent=4, separators=(',', ':'))


class ModelExtractor:

    def __init__(self, modelPath):
        self.layerMethods = {
            'Conv2D': self.convSummary,
            'Conv1D': self.convSummary,
            'Conv3D': self.convSummary,
            'InputLayer': self.inputSummary

        }

        self.model = load_model(modelPath)
        self.summary = self.extract(self.model)

    def extract(self, model) -> ModelSummary:
        modelSummary = ModelSummary()
        modelSummary.class_name = model.__class__.__name__
        modelSummary.name = model.name

        if modelSummary.class_name == 'Sequential':
            summary = LayerSummary()
            summary.class_name = 'InputLayer'
            summary.name = 'inputLayer'
            summary.shape = model.layers[0].input_shape[1:]
            summary.neurons = reduce(lambda x, y: x*y, summary.shape)
            modelSummary.layers.append(summary)

        for layer in model.layers:
            if layer.__class__.__name__ in ['Sequential', 'Functional']:
                summary = self.extract()

            else:
                summary = LayerSummary()
                if layer.__class__.__name__ in self.layerMethods:
                    summary = self.layerMethods[layer.__class__.__name__](
                        layer)
            modelSummary.neurons += summary.neurons
            modelSummary.additions += summary.additions
            modelSummary.multiplications += summary.multiplications
            modelSummary.comparisions += summary.comparisions
            modelSummary.connections += summary.connections
            modelSummary.layers.append(summary)

        return modelSummary

    def inputSummary(self, layer):
        lsummary = LayerSummary()

        lsummary.class_name = layer.__class__.__name__
        lsummary.name = layer.name

        # lsummary.shape

        if(type(layer.output_shape) == type([])):
            lsummary.shape = [eachoutput[1:]
                              for eachoutput in layer.output_shape]
            lsummary.neurons = sum(
                [reduce(lambda x, y: x*y, lsummary.shape[i]) for i in range(len(lsummary.shape))])
        else:
            lsummary.shape = layer.output_shape[1:]
            lsummary.neurons = reduce(lambda x, y: x*y, lsummary.shape)

        return lsummary

    def convSummary(self, layer):
        # We can formulate this and improve the speed
        lsummary = LayerSummary()

        lsummary.class_name = layer.__class__.__name__
        lsummary.name = layer.name

        if(type(layer.output_shape) == type([])):
            lsummary.shape = [eachoutput[1:]
                              for eachoutput in layer.output_shape]
            lsummary.neurons = sum(
                [reduce(lambda x, y: x*y, lsummary.shape[i]) for i in range(len(lsummary.shape))])
        else:
            lsummary.shape = layer.output_shape[1:]
            lsummary.neurons = reduce(lambda x, y: x*y, lsummary.shape)

        layercopy = deepcopy(layer)

        # no of additions will be equal to number of multiplications in conv layers
        layercopy.set_weights(
            [np.ones(layer.weights[0].shape), np.zeros(layer.weights[1].shape)])

        lsummary.additions = int(tf.math.reduce_sum(
            layercopy(np.ones((1, *layer.input_shape[1:])))))
        lsummary.multiplications = lsummary.additions

        lsummary.connections = lsummary.neurons * \
            reduce(lambda x, y: x*y, layer.kernel_size)
        lsummary.comparisions = 0

        return lsummary

    '''
    def func(layer):
        lsummary = LayerSummary()

        inputShape = layer.input_shape

        lsummary.class_name = layer.__class__.__name__
        lsummary.name = layer.name
        if(type(layer.output_shape) == 'list'):
            lsummary.shape = [eachoutput[1:]
                              for eachoutput in layer.output_shape]
        else:
            lsummary.shape = layer.output_shape[1:]
        lsummary.additions = 
        lsummary.multiplications = 
        lsummary.connections = 
        lsummary.comparasions = 
        lsummary.neurons = sum(
            [reduce(lambda x, y: x*y, lsummary.shape[i]) for i in range(len(lsummary.shape))])


        return lsummary
    
    '''
