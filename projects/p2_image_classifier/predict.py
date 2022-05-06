import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json

from PIL import Image

import argparse

parser = argparse.ArgumentParser(description='Uses a trained network to predict the image')
parser.add_argument('image_file', help='Image file to predict')
parser.add_argument('saved_model', help='Saved trained model')

# optional argument
parser.add_argument('--top_k', type=int, metavar='', help='Top K most likely classes')
parser.add_argument('--category_names', metavar='', help='Path to a JSON file mapping labels to flower names.')

args = parser.parse_args()

def process_image(image):
	image_size = 224
	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, (image_size, image_size))
	image /= 255
	return image.numpy().squeeze()

def original_image(img_path):
    im = Image.open(img_path)
    #im = Image.fromarray(img_path)
    test_image = np.asarray(im)
    return test_image

def processed_image(img_path):
    
    im = Image.open(img_path)
    #im = Image.fromarray(img_path)
    test_image = np.asarray(im)
    return process_image(test_image)

def mypredict(img_path, model, top_k):
    
    test_image = original_image(img_path)
    processed_test_image = process_image(test_image)
    new_image = np.expand_dims(processed_test_image, axis=0)
    
    ps = model.predict(new_image)

    sorted_ps = -np.sort(-ps)  # descending order
    sorted_ps_index = np.argsort(-ps)+1  # label needs to be shifted.

    return sorted_ps[:,:top_k], sorted_ps_index[:,:top_k]

if __name__ == '__main__':

	loaded_keras_model = tf.keras.models.load_model(args.saved_model, custom_objects={'KerasLayer':hub.KerasLayer})
	#loaded_keras_model.summary()

	image_file = args.image_file

	top_k = 1
	if args.top_k:
		top_k = args.top_k

	probs, classes = mypredict(image_file, loaded_keras_model, top_k)

	print("Top ", top_k, " probability: ", probs)
	print("Top ", top_k, " classes: ", classes)

	if args.category_names:
		with open(args.category_names, 'r') as f:
			class_names = json.load(f)

		flower_names = []
		for i in range(top_k):
			flower_names.append(class_names[str(classes[0][i])])
		print("Top ", top_k, " most likely image: ", flower_names)
