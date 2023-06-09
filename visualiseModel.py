import os
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import load_model

# Get the models
models = []
for filename in os.listdir('Models'):
    if filename.endswith('.h5'):
        print('Loading model: ' + filename)
        try:
            model = load_model(os.path.join('Models', filename))
            models.append(model)
        except:
            print('Failed to load model: ' + filename)

# Create a directory for the images if it doesn't exist
if not os.path.exists('ModelImages'):
    os.makedirs('ModelImages')
else:
    # Delete all the images in the directory
    for filename in os.listdir('ModelImages'):
        os.remove(os.path.join('ModelImages', filename))

# Plot the models
for model in models:
    plot_model(model, to_file=os.path.join('ModelImages', model.name + '.png'), show_shapes=True)

