from keras.layers import *
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os
from functools import reduce
import numpy as np
from copy import deepcopy


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

    def __dict__(self) -> dict:
        return{
            "class_name": self.class_name,
            "name": self.name,
            "layers": [layer.__dict__() for layer in self.layers],
            "neurons": self.neurons,
            "additions": self.additions,
            "multiplications": self.multiplications,
            "comparasions": self.comparisions,
            "connections": self.connections,
        }

    def __repr__(self):
        return json.dumps({
            "class_name": self.class_name,
            "name": self.name,
            "layers": [layer.__dict__() for layer in self.layers],
            "neurons": self.neurons,
            "additions": self.additions,
            "multiplications": self.multiplications,
            "comparasions": self.comparisions,
            "connections": self.connections,
        }, indent=4)


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

    def __dict__(self) -> dict:
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
        }, indent=4)


class ModelExtractor:

    def __init__(self, modelPath):
        self.layerMethods = {
            'InputLayer': self.inputSummary,
            'Dense': self.denseSummary,
            # 'Flatten': self.flattenSummary,
            'Conv1D': self.conv1DSummary,
            'Conv2D': self.conv2DSummary,
            'Conv3D': self.conv3DSummary,
            # 'MaxPooling1D': self.maxPooling1DSummary,
            # 'MaxPooling2D': self.maxPooling2DSummary,
            # 'MaxPooling3D': self.maxPooling3DSummary,
            # 'Average': self.averageSummary,
            # 'Add': self.addSummary,
            # 'Subtract': self.subtractSummary,
            # 'Dot': self.dotSummary,
            # 'Minimum': self.minimumSummary,
            # 'Maximum': self.maximumSummary,
            # 'Rescaling': self.rescalingSummary,
            # 'Resizing': self.resizingSummary

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
                summary = self.extract(layer)

            else:
                summary = LayerSummary()
                summary.class_name = layer.__class__.__name__
                summary.name = layer.name
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
                              for eachoutput in layer.output_shape]  # [(2,3),(2,3)]
            lsummary.neurons = sum(
                [reduce(lambda x, y: x*y, lsummary.shape[i]) for i in range(len(lsummary.shape))])
        else:
            lsummary.shape = layer.output_shape[1:]
            lsummary.neurons = reduce(lambda x, y: x*y, lsummary.shape)

        return lsummary

    def conv1DSummary(self, layer):

        config = layer.get_config()
        lsummary = LayerSummary()
        lsummary.class_name = layer.__class__.__name__
        lsummary.name = layer.name

        lsummary.shape = layer.output_shape[1:]
        lsummary.neurons = reduce(lambda x, y: x*y, lsummary.shape)

        channels = layer.input_shape[-2] if config['data_format'] == 'channels_first' else layer.input_shape[-1]
        kernel_size = config['kernel_size'][0]
        lsummary.additions = lsummary.neurons*channels * \
            kernel_size if config['use_bias'] else (
                lsummary.neurons-1)*channels*kernel_size
        lsummary.multiplications = lsummary.neurons*channels*kernel_size
        lsummary.connections = lsummary.neurons*channels*kernel_size

        return lsummary

    def conv2DSummary(self, layer: Conv2D) -> LayerSummary:
        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[-3:]
        summary.neurons = reduce(lambda x, y: x*y, summary.shape)
        # finding number of additions
        config = layer.get_config()
        channels = layer.input_shape[-1 if config['data_format']
                                     == 'channels_last' else -3]
        kernel_nodes = reduce(lambda x, y: x*y, config['kernel_size'])
        summary.additions = summary.neurons*kernel_nodes*channels
        if config['use_bias'] == False:
            summary.additions -= summary.neurons
        summary.multiplications = summary.neurons*kernel_nodes*channels
        summary.connections = summary.neurons*kernel_nodes*channels

        return summary

    def conv3DSummary(self, layer: Conv3D) -> LayerSummary:
        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[-4:]
        summary.neurons = reduce(lambda x, y: x*y, summary.shape)
        config = layer.get_config()
        channels = layer.input_shape[-1 if config['data_format']
                                     == 'channels_last' else -4]
        kernel_nodes = reduce(lambda x, y: x*y, config['kernel_size'])
        summary.additions = summary.neurons*kernel_nodes*channels
        if config['use_bias'] == False:
            summary.additions -= summary.neurons
        summary.multiplications = summary.neurons*kernel_nodes*channels
        summary.connections = summary.neurons*kernel_nodes*channels

    def denseSummary(self, layer):
        lsummary = LayerSummary()
        lsummary.class_name = layer.__class__.__name__
        lsummary.name = layer.name
        lsummary.shape = layer.output_shape[1:]
        lsummary.neurons = layer.get_config['units']
        lsummary.additions = layer.input_shape[-1] * \
            lsummary.neurons if layer.get_config['use_bias'] else layer.input_shape[-1]*(
                lsummary.neurons-1)
        lsummary.multiplications = layer.input_shape[-1]*lsummary.neurons
        lsummary.connections = layer.input_shape[-1]*lsummary.neurons

        return lsummary

    def maxPooling1DSummary(self, layer):
        pass

    def maxPooling2DSummary(self, layer):
        pass

    def maxPooling3DSummary(self, layer):
        pass

    def averageSummary(self, layer):
        pass

    def addSummary(self, layer):
        pass

    def subtractSummary(self, layer):
        pass

    def dotSummary(self, layer):
        pass

    def minimumSummary(self, layer):
        pass

    def maximumSummary(self, layer):
        pass

    def flattenSummary(self, layer):
        pass

    def rescalingSummary(self, layer):
        pass

    def resizingSummary(self, layer):
        pass

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


if __name__ == '__main__':
    ext = ModelExtractor('Models/Conv1DTest.h5')
    print(ext.summary)

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
