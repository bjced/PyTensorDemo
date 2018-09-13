import tensorflow as tf
import os
from PIL import Image
import numpy as np
import cv2
import sys

graph_def = tf.GraphDef()
labels = []

def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image

def crop_center(img,cropx,cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

def resize_down_to_max_dim(image, max_dim):
    h, w = image.shape[:2]
    if (h < max_dim and w < max_dim):
        return image

    new_size = (max_dim * w // h, max_dim) if (h > w) else (max_dim, max_dim * h // w)
    return cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)

def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)

# Import the TF graph
with tf.gfile.FastGFile("model.pb", 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# Create a list of labels.
with open("labels.txt", 'rt') as lf:
    for l in lf:
        labels.append(l.strip())

# Load from a file
imageFile = "banana.jpg"
if len(sys.argv) > 1:
     imageFile = sys.argv[1]
image = Image.open(imageFile)

# Convert to OpenCV format
image = convert_to_opencv(image)

# If the image has either w or h greater than 1600 we resize it down respecting
# aspect ratio such that the largest dimension is 1600
augmented_image = resize_down_to_max_dim(image, 1600)

# We next get the largest center square
h, w = image.shape[:2]
min_dim = min(w,h)
augmented_image = crop_center(image, min_dim, min_dim)

# Resize that square down to 256x256
augmented_image = resize_to_256_square(augmented_image)

# The compact models have a network size of 227x227, the model requires this size.
network_input_size = 227

# Crop the center for the specified network_input_Size
augmented_image = crop_center(augmented_image, network_input_size, network_input_size)

cv2.imshow('img', augmented_image)
cv2.imwrite(sys.argv[2], augmented_image)
#cv2.waitKey(1500)
# These names are part of the model and cannot be changed.
#output_layer = 'loss:0'
#input_node = 'Placeholder:0'
#
#with tf.Session() as sess:
#    prob_tensor = sess.graph.get_tensor_by_name(output_layer)
#    predictions, = sess.run(prob_tensor, {input_node: [augmented_image] })
#
#
## Print the highest probability label
#    highest_probability_index = np.argmax(predictions)
#    print('Classified as: ' + labels[highest_probability_index])
#    print()
#
#    # Or you can print out all of the results mapping labels to probabilities.
#    label_index = 0
#    for p in predictions:
#        truncated_probablity = np.float64(round(p,8))
#        print (labels[label_index], truncated_probablity)
#        label_index += 1