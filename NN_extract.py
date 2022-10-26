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

# Special Layers
MPOOL_LAYERS = ['MaxPooling1D','MaxPooling2D', 'MaxPooling3D']
APOOL_LAYERS = ['AveragePooling1D', 'AveragePooling2D', 'AveragePooling3D']
NPOOL_LAYERS = MPOOL_LAYERS + APOOL_LAYERS
GMPOOL_LAYERS = ['GlobalMaxPooling1D','GlobalMaxPooling2D', 'GlobalMaxPooling3D']
GAPOOL_LAYERS = ['GlobalAveragePooling1D', 'GlobalAveragePooling2D', 'GlobalAveragePooling3D']
GPOOL_LAYERS = GMPOOL_LAYERS + GAPOOL_LAYERS
POOL_LAYERS = NPOOL_LAYERS + GPOOL_LAYERS

class Model_Extractor:

    def __init__(self, model_json_str, output_filename = 'output.json'):
        self.model = model_from_json(model_json_str)
        self.model_json = json.loads(model_json_str)
        self.config = self.model_json['config']
        self.outputs = dict()

        self.make_target_layer_models()
        
        self.get_output()
        self.dump_output(output_filename)
        
        
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
        
        if "activation" in layer_model_json['config']['layers'][1]['config']:
            layer_model_json['config']['layers'][1]['config']['activation'] = 'relu'

        layer_model = model_from_json(json.dumps(layer_model_json))
        return layer_model
    
    def make_target_layer_models(self):
        target_layer_models = {}
        for layer in self.config['layers'][1:]:
            target_layer_models[layer['config']['name']] = self.make_model_of_layer(layer)
        self.target_layer_models = target_layer_models
    
    def get_layer_connections(self,target_layer_json: json,layer_i: int):
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
        if target_layer_json['class_name'] in NPOOL_LAYERS:
            pool_size = np.prod(target_layer_json['config']['pool_size'])
            no_of_neurons = self.outputs["layers_ls"][layer_i]["Neuron_count"]
            return no_of_neurons*pool_size
        if target_layer_json['class_name'] in GPOOL_LAYERS:
            prev_layer_shape = self.outputs['layers_ls'][layer_i-1]['shape']
            no_of_connections = int(np.prod(prev_layer_shape))
            return no_of_connections
        return None

    def get_connections(self):
        self.outputs['layers_ls'][0]['Number of connections'] = None
        total_connections = 0
        for layer_i, (layer, out_layer) in enumerate(zip(self.config['layers'][1:],self.outputs['layers_ls'][1:]),start=1):
            out_layer['Number of connections'] = np.sum(self.get_layer_connections(layer,layer_i))
            if out_layer['Number of connections']:
                out_layer['Number of connections'] = int(out_layer['Number of connections'])
                total_connections += out_layer['Number of connections']
        self.outputs['Total number of connections'] = total_connections
    
    def get_layer_multiplications(self, target_layer_json: json, layer_i: int):
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
            return int(np.sum(features[0]))
        if target_layer_json['class_name'] in APOOL_LAYERS + GAPOOL_LAYERS:
            return 1
        return 0

    def get_multiplications(self):
        self.outputs['layers_ls'][0]['Number of multiplications'] = 0
        total_multiplications = 0
        total_divisions = 0
        for layer_i, (layer, out_layer) in enumerate(zip(self.config['layers'][1:],self.outputs['layers_ls'][1:]),start=1):
            # out_layer['Layer connections'] = self.get_layer_connections(layer)
            if layer['class_name'] in APOOL_LAYERS + GAPOOL_LAYERS:
                # Returns number of divisions in AveragePool layers
                out_layer['Number of divisions'] = self.get_layer_multiplications(layer,layer_i)
                total_divisions += out_layer['Number of divisions']

            else:
                out_layer['Number of multiplications'] = self.get_layer_multiplications(layer,layer_i)
                total_multiplications += out_layer['Number of multiplications']
        self.outputs['Total number of multiplications'] = total_multiplications
    
    def get_layer_additions(self,target_layer_json: json, layer_i):
        layer_model = self.target_layer_models[target_layer_json['config']['name']]
        cur_layer_output = self.outputs['layers_ls'][layer_i]
        input_shape = layer_model.layers[0].input_shape
        if layer_model.get_weights():
            weights_shape, biases_shape = layer_model.get_weights()
            weights_shape, biases_shape = weights_shape.shape, biases_shape.shape
            weights = np.zeros(weights_shape)
            biases = np.zeros(biases_shape) + 1
            layer_model.set_weights([weights,biases])
            extractor = tf.keras.Model(inputs=layer_model.inputs,outputs=layer_model.layers[0].output)

            features = extractor(np.zeros((1, *input_shape[1:])))
            return int(np.sum(features[0]))
        if target_layer_json['class_name'] in NPOOL_LAYERS:
            # Returns the number of additions for AveragePool layers and number of comparisions in MaxPool layers
            pool_size = int(np.prod(target_layer_json['config']['pool_size']))
            adds_per_neuron = pool_size - 1
            no_of_neurons = cur_layer_output["Neuron_count"]
            return adds_per_neuron*no_of_neurons
        if target_layer_json['class_name'] in GPOOL_LAYERS:
            # Number of additions or comparision in Global pool layers = Number of connection - Number of neurons
            return cur_layer_output["Number of connections"] - cur_layer_output["Neuron_count"]
        return 0
        
    def get_additions(self):
        self.outputs['layers_ls'][0]['Number of additions'] = 0
        total_additions = 0
        total_comparisions = 0
        # for layer, out_layer in zip(self.config['layers'][1:],self.outputs['layers_ls'][1:]):
        for layer_i, (layer, out_layer) in enumerate(zip(self.config['layers'][1:],self.outputs['layers_ls'][1:]),start=1):
            # out_layer['Layer connections'] = self.get_layer_connections(layer)
            if layer['class_name'] in MPOOL_LAYERS + GMPOOL_LAYERS:
                # Records number of comparisions for MaxPool layers
                out_layer['Number of comparisions'] = self.get_layer_additions(layer,layer_i)
                total_comparisions += out_layer['Number of comparisions']
            else:
                # Records number of additions for AveragePool layers
                out_layer['Number of additions'] = self.get_layer_additions(layer, layer_i)
                total_additions += out_layer['Number of additions']
        self.outputs['Total number of additions'] = total_additions
        self.outputs['Total number of comparisions'] = total_comparisions

    def get_output(self):
        self.get_layers()
        # print(self)
        self.get_layer_shape()
        self.get_neurons()
        self.get_connections()
        self.get_multiplications()
        self.get_additions()

    def dump_output(self,filename):
        # Saves the output in a given file
        with open(filename,'w') as output_file:
            json.dump((self.outputs),output_file,indent=4, separators=(',',' : '))
        print(f'\nSuccessfully dumped outputs to {filename}')
    
def main():
    # MNIST_convnet_extractor = Model_Extractor(models_json_str['Simple_MNIST_convnet'])
    model_extractor = Model_Extractor(models_json_str['LSTMJson'])
    # print(MNIST_convnet_extractor.outputs)
    # print(model_extractor.model.layers[0].dtype)

        # print((MNIST_convnet_extractor.outputs))
        # MNIST_convnet_extractor.outputs

if __name__ == "__main__":
    main()