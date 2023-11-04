from keras.layers import *
import tensorflow as tf
from keras.models import load_model
import json
import os
from functools import reduce
import numpy as np
import shutil
import csv
import sys

modelPaths = dict()
for filepath in os.listdir('Models'):
    filename, ext = os.path.splitext(filepath)

    if ext in ['.h5']:
        modelPaths[filename] = os.path.join('Models', filepath)


class ModelSummary:

    def __init__(self) -> None:
        self.class_name = ''
        self.name = ''
        self.layers = []
        self.neurons = 0
        self.additions = 0
        self.multiplications = 0
        self.divisions = 0
        self.comparisions = 0
        self.connections = 0
        self.weights = 0
        self.biases = 0

    def __dict__(self) -> dict:
        return{
            "class_name": self.class_name,
            "name": self.name,
            "layers": [layer.__dict__() for layer in self.layers],
            "neurons": self.neurons,
            "additions": self.additions,
            "multiplications": self.multiplications,
            "divisions": self.divisions,
            "comparasions": self.comparisions,
            "connections": self.connections,
            "weights": self.weights,
            "biases": self.biases,
        }

    def __repr__(self):
        return json.dumps({
            "class_name": self.class_name,
            "name": self.name,
            "layers": [layer.__dict__() for layer in self.layers],
            "neurons": self.neurons,
            "additions": self.additions,
            "multiplications": self.multiplications,
            "divisions": self.divisions,
            "comparasions": self.comparisions,
            "connections": self.connections,
            "weights": self.weights,
            "biases": self.biases,
        }, indent=4)

    def save_as_csv(self):
        #check if the dir "export_csv" exists
        csvFileName = self.name + '.csv'
        dirpath = os.path.join(os.path.dirname(__file__), "export_csv")
        if(not os.path.exists(dirpath)):
            os.mkdir(dirpath)
        with open(os.path.join(dirpath, csvFileName), 'w') as f:    
            writer = csv.writer(f)
            header = ['Layer_no', 'Layer_class','Layer_name', 'Neurons', 'Additions', 'Multiplications', 'Divisions', 'Comparasions', 'Connections']
            writer.writerow(header)
            for i,layer in enumerate(self.layers):
                if(layer.__class__.__name__ == 'ModelSummary'):
                    layer.save_as_csv()
                row = [i, layer.class_name, layer.name, layer.neurons, layer.additions, layer.multiplications, layer.divisions, layer.comparisions, layer.connections]
                writer.writerow(row)
            f.close()

class LayerSummary:
    def __init__(self) -> None:
        self.class_name = ''
        self.name = ''
        self.shape = []
        self.neurons = 0
        self.additions = 0
        self.multiplications = 0
        self.divisions = 0
        self.comparisions = 0
        self.connections = 0
        self.weights = 0
        self.biases = 0

    def __dict__(self) -> dict:
        return {
            "class_name": self.class_name,
            "name": self.name,
            "shape": self.shape,
            "neurons": self.neurons,
            "additions": self.additions,
            "multiplications": self.multiplications,
            "divisions": self.divisions,
            "comparasions": self.comparisions,
            "connections": self.connections,
            "weights": self.weights,
            "biases": self.biases,
        }

    def __repr__(self):
        return json.dumps({
            "class_name": self.class_name,
            "name": self.name,
            "shape": self.shape,
            "neurons": self.neurons,
            "additions": self.additions,
            "multiplications": self.multiplications,
            "divisions": self.divisions,
            "comparasions": self.comparisions,
            "connections": self.connections,
            "weights": self.weights,
            "biases": self.biases,
        }, indent=4)


class ModelExtractor:

    def __init__(self, modelPath):
        self.layerMethods = {
            'InputLayer': self.inputSummary,
            'Flatten': self.flattenSummary,
            'Dense': self.denseSummary,
            'Conv1D': self.conv1DSummary,
            'Conv2D': self.conv2DSummary,
            'Conv3D': self.conv3DSummary,
            'MaxPooling1D': self.maxPooling1DSummary,
            'MaxPooling2D': self.maxPooling2DSummary,
            'MaxPooling3D': self.maxPooling3DSummary,
            'AveragePooling1D': self.averagePooling1DSummary,
            'AveragePooling2D': self.averagePooling2DSummary,
            'AveragePooling3D': self.averagePooling3DSummary,
            'GlobalMaxPooling1D': self.globalMaxPooling1DSummary,
            'GlobalMaxPooling2D': self.globalMaxPooling2DSummary,
            'GlobalMaxPooling3D': self.globalMaxPooling3DSummary,
            'GlobalAveragePooling1D': self.globalAveragePooling1DSummary,
            'GlobalAveragePooling2D': self.globalAveragePooling2DSummary,
            'GlobalAveragePooling3D': self.globalAveragePooling3DSummary,
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

    def to_json(self,filepath='output.json'):
        with open(filepath,'w') as f:
            f.write(str(self.summary))
    

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
                    summary = self.layerMethods[layer.__class__.__name__](layer)
            modelSummary.neurons += summary.neurons
            modelSummary.additions += summary.additions
            modelSummary.multiplications += summary.multiplications
            modelSummary.divisions += summary.divisions
            modelSummary.comparisions += summary.comparisions
            modelSummary.connections += summary.connections
            modelSummary.weights += summary.weights
            modelSummary.biases += summary.biases
            modelSummary.layers.append(summary)

        return modelSummary



    def inputSummary(self, layer: InputLayer) -> LayerSummary:
        summary = LayerSummary()

        summary.class_name = layer.__class__.__name__
        summary.name = layer.name

        # summary.shape

        if(type(layer.output_shape) == type([])):
            summary.shape = [eachoutput[1:]for eachoutput in layer.output_shape]
            summary.neurons = sum([reduce(lambda x, y: x*y, summary.shape[i]) for i in range(len(summary.shape))])
        else:
            summary.shape = layer.output_shape[1:]
            summary.neurons = reduce(lambda x, y: x*y, summary.shape)

        return summary

    def conv1DSummary(self, layer: Conv1D) -> LayerSummary:

        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name

        summary.shape = layer.output_shape[1:]
        summary.neurons = reduce(lambda x, y: x*y, summary.shape)

        channels = layer.input_shape[-2] if layer.data_format =='channels_first' else layer.input_shape[-1]
        kernel_size = layer.kernel_size[0]
        summary.additions = summary.neurons*channels * kernel_size if layer.use_bias else (summary.neurons-1)*channels*kernel_size
        summary.multiplications = summary.neurons*channels*kernel_size
        summary.connections = summary.neurons*channels*kernel_size
        summary.weights = int(tf.size(layer.weights[0]))
        if layer.use_bias:
            summary.biases = int(tf.size(layer.weights[1]))

        return summary

    def conv2DSummary(self, layer: Conv2D) -> LayerSummary:
        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[-3:]
        summary.neurons = reduce(lambda x, y: x*y, summary.shape)
        # finding number of additions
        channels = layer.input_shape[-1 if layer.data_format == 'channels_last' else -3]
        kernel_nodes = reduce(lambda x, y: x*y, layer.kernel_size)
        summary.additions = summary.neurons*kernel_nodes*channels
        if layer.use_bias == False:
            summary.additions -= summary.neurons
        summary.multiplications = summary.neurons*kernel_nodes*channels
        summary.connections = summary.neurons*kernel_nodes*channels
        summary.weights = int(tf.size(layer.weights[0]))
        if layer.use_bias:
            summary.biases = int(tf.size(layer.weights[1]))

        return summary

    def conv3DSummary(self, layer: Conv3D) -> LayerSummary:
        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[-4:]
        summary.neurons = reduce(lambda x, y: x*y, summary.shape)
        channels = layer.input_shape[-1 if layer.data_format == 'channels_last' else -4]
        kernel_nodes = reduce(lambda x, y: x*y, layer.kernel_size)
        summary.additions = summary.neurons*kernel_nodes*channels
        if layer.use_bias == False:
            summary.additions -= summary.neurons
        summary.multiplications = summary.neurons*kernel_nodes*channels
        summary.connections = summary.neurons*kernel_nodes*channels
        summary.weights = int(tf.size(layer.weights[0]))
        if layer.use_bias:
            summary.biases = int(tf.size(layer.weights[1]))

    def denseSummary(self, layer: Dense) -> LayerSummary:
        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[1:]
        summary.neurons = reduce(lambda x,y:x*y, summary.shape)
        summary.additions = layer.input_shape[-1] * summary.neurons if layer.use_bias else layer.input_shape[-1]*(summary.neurons-1)
        summary.multiplications = layer.input_shape[-1]*summary.neurons
        summary.connections = layer.input_shape[-1]*summary.neurons
        summary.weights = int(tf.size(layer.weights[0]))
        if layer.use_bias:
            summary.biases = int(tf.size(layer.weights[1]))

        return summary

    def maxPooling1DSummary(self, layer: MaxPooling1D) -> LayerSummary:

        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[1:]
        summary.neurons = reduce(lambda x, y: x*y, summary.shape)
        summary.connections = summary.neurons*layer.pool_size
        summary.comparisions = (layer.pool_size-1)*summary.neurons
        return summary

    def maxPooling2DSummary(self, layer: MaxPooling2D)->LayerSummary:
        
        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[1:]
        summary.neurons = reduce(lambda x,y: x*y, summary.shape)
        summary.connections = summary.neurons*reduce(lambda x, y: x*y, layer.pool_size)
        summary.comparisions = summary.neurons*(reduce(lambda x,y: x*y, layer.pool_size)-1)
        return summary

    def maxPooling3DSummary(self, layer: MaxPooling3D)-> LayerSummary: 
        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[1:]
        summary.neurons = reduce(lambda x,y: x*y, summary.shape)
        summary.connections = summary.neurons*reduce(lambda x,y:x*y, layer.pool_size)
        summary.comparisions = summary.neurons*(reduce(lambda x,y: x*y, layer.pool_size)-1)
        return summary

    def averagePooling1DSummary(self, layer:AveragePooling1D) -> LayerSummary:
        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[1:]
        summary.neurons = reduce(lambda x, y: x*y, summary.shape)
        summary.additions = summary.neurons*(layer.pool_size-1)
        summary.connections = summary.neurons*layer.pool_size
        summary.divisions = summary.neurons

        return summary
    
    def averagePooling2DSummary(self,layer: AveragePooling2D) -> LayerSummary:
        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[1:]
        summary.neurons = reduce(lambda x, y: x*y, summary.shape)
        pool_nodes = reduce(lambda x, y: x*y, layer.pool_size)
        summary.additions = summary.neurons*(pool_nodes-1)
        summary.connections = summary.neurons*(pool_nodes)
        summary.divisions = summary.neurons

        return summary
        
    def averagePooling3DSummary(self,layer: AveragePooling3D) -> LayerSummary:
        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[1:]
        summary.neurons = reduce(lambda x, y: x*y, summary.shape)
        pool_nodes = reduce(lambda x, y: x*y, layer.pool_size)
        summary.additions = summary.neurons*(pool_nodes-1)
        summary.connections = summary.neurons*(pool_nodes)
        summary.divisions = summary.neurons
        
        return summary
    
    def globalMaxPooling1DSummary(self, layer: GlobalMaxPooling1D)-> LayerSummary:

        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[1:]
        summary.neurons = reduce(lambda x, y: x*y, summary.shape)
        summary.connections = reduce(lambda x,y: x*y, layer.input_shape[1:])
        summary.comparisions = summary.connections - summary.neurons
        return summary

    def globalMaxPooling2DSummary(self, layer: GlobalMaxPooling2D)-> LayerSummary:
        
        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[1:]
        summary.neurons = reduce(lambda x,y: x*y, summary.shape)
        summary.connections = reduce(lambda x,y:x*y, layer.input_shape[1:])
        summary.comparisions = summary.connections - summary.neurons
        return summary
    
    def globalMaxPooling3DSummary(self, layer: GlobalMaxPooling3D)-> LayerSummary:
        
        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[1:]
        summary.neurons = reduce(lambda x,y: x*y, summary.shape)
        summary.connections = reduce(lambda x,y:x*y, layer.input_shape[1:])
        summary.comparisions = summary.connections - summary.neurons
        return summary
        
    def globalAveragePooling1DSummary(self,layer: GlobalAveragePooling1D) -> LayerSummary:
        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[1:]
        summary.neurons = reduce(lambda x, y: x*y, summary.shape)
        data_format = layer.data_format
        steps = layer.input_shape[1 if data_format == 'channels_last' else 2]
        summary.additions = summary.neurons*(steps-1)
        summary.connections = summary.neurons*steps
        summary.divisions = summary.neurons

        return summary

    def globalAveragePooling2DSummary(self,layer: GlobalAveragePooling2D) -> LayerSummary:
        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[1:]
        summary.neurons = reduce(lambda x, y: x*y, summary.shape)
        data_format = layer.data_format
        if data_format == 'channels_last':
            input_dim = layer.input_shape[1:3]
        else:
            input_dim = layer.input_shape[2:4]
        input_nodes = input_dim[0]*input_dim[1]
        summary.additions = summary.neurons*(input_nodes-1)
        summary.connections = summary.neurons*input_nodes
        summary.divisions = summary.neurons

        return summary

    def globalAveragePooling3DSummary(self,layer: GlobalAveragePooling3D) -> LayerSummary:
        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[1:]
        summary.neurons = reduce(lambda x, y: x*y, summary.shape)
        data_format = layer.data_format
        if data_format == 'channels_last':
            input_dim = layer.input_shape[1:3]
        else:
            input_dim = layer.input_shape[2:4]
        input_nodes = input_dim[0]*input_dim[1]*input_dim[2]
        summary.additions = summary.neurons*(input_nodes-1)
        summary.connections = summary.neurons*input_nodes
        summary.divisions = summary.neurons

        return summary

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

    def flattenSummary(self, layer:Flatten) -> LayerSummary:
        summary = LayerSummary()
        summary.class_name = layer.__class__.__name__
        summary.name = layer.name
        summary.shape = layer.output_shape[1:]
        summary.neurons = summary.shape[0]
        summary.connections = summary.neurons
        
        return summary

    def rescalingSummary(self, layer):
        pass

    def resizingSummary(self, layer):
        pass


if __name__ == '__main__':
    exportdirpath = os.path.join(os.path.dirname(__file__), 'export_csv')
    if(os.path.exists(exportdirpath)):
        shutil.rmtree(exportdirpath)

    for model in os.listdir('Models'):
        modelPath = os.path.join('Models', model)
        # print(modelPath)
        try:
            ext = ModelExtractor(modelPath)
            summary = ext.summary
            summary.save_as_csv()
        except:
            print('Error in model: ', modelPath)
            print('Error: ', sys.exc_info()[0])
            continue