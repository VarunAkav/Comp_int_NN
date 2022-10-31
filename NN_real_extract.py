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

    def toJson(self):
        return {
            "class_name": self.class_name,
            "name": self.name,
            "layers": self.layers,
            "neurons": self.neurons,
            "additions": self.additions,
            "multiplications": self.multiplications,
            "comparasions": self.comparisions,
            "connections": self.connections,
        }


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

    def toJson(self):
        return {
            "class_name": self.class_name,
            "name": self.name,
            "shape": self.shape,
            "neurons": self.neurons,
            "additions": self.additions,
            "multiplications": self.multiplications,
            "comparasions": self.comparisions,
            "connections": self.connections,
        }


class ModelExtractor:

    layerMethods = dict()

    def __init__(self, modelPath):
        print('something')
        self.model = load_model(modelPath)
        self.summary = self.extract(self.model)

        # dict should contain methods which will return LayerSummary()

    def extract(self, model) -> ModelSummary:
        modelSummary = ModelSummary()
        modelSummary.class_name = model.__class__.__name__
        modelSummary.name = model.name

        for layer in model.layers:
            if layer.__class__.__name__ in ['Sequential', 'Functional']:
                summary = self.extract()

            else:
                summary = LayerSummary()
                if layer.__class__.__name__ in self.layerMethods:
                    summary = self.layerMethods[layer.__class__.__name__](
                        layer)

            modelSummary.additions += summary.additions
            modelSummary.multiplications += summary.multiplications
            modelSummary.comparisions += summary.comparisions
            modelSummary.connections += summary.connections
            modelSummary.layers.append(summary)

        return modelSummary

    def InputLayer(self, layer):
        lsummary = LayerSummary()

        lsummary.class_name = layer.__class__.__name__
        lsummary.name = layer.name
        if(type(layer.output_shape) == 'list'):
            lsummary.shape = [eachoutput[1:]
                              for eachoutput in layer.output_shape]
        else:
            lsummary.shape = layer.output_shape[1:]
        lsummary.neurons = sum(
            [reduce(lambda x, y: x*y, lsummary.shape[i]) for i in range(len(lsummary.shape))])

        return lsummary

    '''
    def func(layer):
        lsummary = LayerSummary()

        inputShape = layer.input_shape

        lsummary.class_name = layer.__class__.__name__
        lsummary.name = layer.name
        lsummary.shape = layer.output_shape[1:]
        lsummary.additions = 
        lsummary.multiplications = 
        lsummary.connections = 
        lsummary.comparasions = 
        lsummary.neurons = reduce(lambda x, y: x*y, lsummary.shape)


        return lsummary
    
    '''
