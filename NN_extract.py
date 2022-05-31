import tensorflow as tf
from tensorflow.keras.models import model_from_json
import json
import os
from functools import reduce
import numpy as np
from copy import deepcopy

# All the models from the Models folder will be stored in a dictionary
models_json_str = dict()
for filepath in os.listdir('Models'):
    with open(os.path.join('Models',filepath),'r') as model_file:
        filename, ext = os.path.splitext(filepath)
        if ext == '.json':
            models_json_str[filename] = model_file.read()
            
class Model_Extract:
    def __init__(self, model_json_str):
        self.model = model_from_json(model_json_str)
        self.model_json = json.loads(model_json_str)
        self.config = self.model_json['config']
        # self.target_layer_models = dict()
        self.outputs = dict()

        self.make_target_layer_models()
        
        self.get_layers()
        self.get_layer_shape()
        self.get_neurons()
        self.get_connections()
        self.get_multiplications()
        self.get_additions()
        
    def get_layers(self):
        self.outputs['layers_ls'] = []
        for layer in self.model_json['config']['layers']:
            self.outputs['layers_ls'].append({'class_name': layer['class_name'],'name': layer['config']['name']})
        
        self.outputs['layers_count'] = len(self.outputs['layers_ls'])
    
    def get_layer_shape(self):
        self.outputs['layers_ls'][0]['shape'] = self.config['layers'][0]['config']['batch_input_shape'][1:]
        for layer in self.outputs['layers_ls'][1:]:
            layer['shape'] = [*self.model.get_layer(layer['name']).output_shape[1:]]

    def get_neurons(self):
        total_neuron_count = 0
        for layer in self.outputs['layers_ls']:
            layer['Neuron_count'] = reduce(lambda x,y : x*y, layer['shape'])
            total_neuron_count += layer['Neuron_count']
        
        self.outputs['Total_neuron_count'] = total_neuron_count

    def make_model_of_layer(self,target_layer_json: json):
        input_shape = [*self.model.get_layer(target_layer_json['config']['name']).input_shape]
        layer_model_json = deepcopy(self.model_json)
        layer_model_json['config']['layers'][0]['config']['batch_input_shape'] = input_shape

        for layer in self.model_json['config']['layers'][1:]:
            if layer == target_layer_json:
                continue
            layer_model_json['config']['layers'].remove(layer)

        layer_model = model_from_json(json.dumps(layer_model_json))
        return layer_model
    
    def make_target_layer_models(self):
        target_layer_models = {}
        for layer in self.config['layers'][1:]:
            target_layer_models[layer['config']['name']] = self.make_model_of_layer(layer)
        self.target_layer_models = target_layer_models
    
    def get_layer_connections(self,target_layer_json: json):
        # layer_model = self.make_model_of_layer(target_layer_json)
        layer_model = self.target_layer_models[target_layer_json['config']['name']]
        input_shape = layer_model.layers[0].input_shape
        if layer_model.get_weights():
            weights_shape, biases_shape = layer_model.get_weights()
            weights_shape, biases_shape = weights_shape.shape, biases_shape.shape
            weights = np.zeros(weights_shape) + 1
            biases = np.zeros(biases_shape)
            layer_model.set_weights([weights,biases])
            extractor = tf.keras.Model(inputs=layer_model.inputs,outputs=layer_model.layers[0].output)

            features = extractor(np.zeros((1, *input_shape[1:]))+1)
            return features[0]
        return None

    def get_connections(self):
        # self.outputs['layers_ls'][0]['Layer connections'] = None
        self.outputs['layers_ls'][0]['Number of connections'] = None
        total_connections = 0
        for layer, out_layer in zip(self.config['layers'][1:],self.outputs['layers_ls'][1:]):
            # out_layer['Layer connections'] = self.get_layer_connections(layer)
            out_layer['Number of connections'] = np.sum(self.get_layer_connections(layer))
            if out_layer['Number of connections']:
                total_connections += out_layer['Number of connections']
        self.outputs['Total number of connections'] = total_connections
    
    def get_connections_per_neuron(self, target_layer_json: json):
        # if target_layer_json['class_name'] == 'Dense':
            pass
    
    def get_layer_multiplications(self, target_layer_json: json):
        layer_model = self.target_layer_models[target_layer_json['config']['name']]
        input_shape = layer_model.layers[0].input_shape
        if layer_model.get_weights():
            weights_shape, biases_shape = layer_model.get_weights()
            weights_shape, biases_shape = weights_shape.shape, biases_shape.shape
            weights = np.zeros(weights_shape) + 1
            biases = np.zeros(biases_shape)
            layer_model.set_weights([weights,biases])
            extractor = tf.keras.Model(inputs=layer_model.inputs,outputs=layer_model.layers[0].output)

            features = extractor(np.zeros((1, *input_shape[1:]))+1)
            return np.sum(features[0])
        return 0

    def get_multiplications(self):
        self.outputs['layers_ls'][0]['Number of multiplications'] = 0
        total_multiplications = 0
        for layer, out_layer in zip(self.config['layers'][1:],self.outputs['layers_ls'][1:]):
            # out_layer['Layer connections'] = self.get_layer_connections(layer)
            out_layer['Number of multiplications'] = self.get_layer_multiplications(layer)
            total_multiplications += out_layer['Number of multiplications']
        self.outputs['Total number of multiplications'] = total_multiplications
    
    def get_layer_additions(self,target_layer_json: json):
        layer_model = self.target_layer_models[target_layer_json['config']['name']]
        input_shape = layer_model.layers[0].input_shape
        if layer_model.get_weights():
            weights_shape, biases_shape = layer_model.get_weights()
            weights_shape, biases_shape = weights_shape.shape, biases_shape.shape
            weights = np.zeros(weights_shape)
            biases = np.zeros(biases_shape) + 1
            layer_model.set_weights([weights,biases])
            extractor = tf.keras.Model(inputs=layer_model.inputs,outputs=layer_model.layers[0].output)

            features = extractor(np.zeros((1, *input_shape[1:])))
            return np.sum(features[0])
        return 0
        
    def get_additions(self):
        self.outputs['layers_ls'][0]['Number of additions'] = 0
        total_additions = 0
        for layer, out_layer in zip(self.config['layers'][1:],self.outputs['layers_ls'][1:]):
            # out_layer['Layer connections'] = self.get_layer_connections(layer)
            out_layer['Number of additions'] = self.get_layer_additions(layer)
            total_additions += out_layer['Number of additions']
        self.outputs['Total number of additions'] = total_additions

def main():

    MNIST_convnet_extract = Model_Extract(models_json_str['Simple_MNIST_convnet'])
    print(json.dumps(MNIST_convnet_extract.outputs, indent=4, separators=('', ' : ')))
    