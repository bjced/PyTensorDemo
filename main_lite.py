import tensorflow as tf
import os
from PIL import Image
import numpy as np
import cv2
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import the tensorflow graph
graph_def = tf.GraphDef()
with tf.gfile.FastGFile("model.pb", 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# Create a list of labels.
labels = []
with open("labels.txt", 'rt') as lf:
    for l in lf:
        labels.append(l.strip())

# Load image
image = Image.open(sys.argv[1])

# Convert to OpenCV format
# RGB -> BGR conversion is performed as well.
image = np.array(np.array(image).T).transpose()

# These names are part of the model and cannot be changed.
output_layer = 'loss:0'
input_node = 'Placeholder:0'

#run inference
with tf.Session() as sess:
    prob_tensor = sess.graph.get_tensor_by_name(output_layer)
    predictions, = sess.run(prob_tensor, {input_node: [image] })


    # Print results
    label_index = 0
    for p in predictions:
        truncated_probablity = np.float64(round(p,8))
        print (labels[label_index], truncated_probablity)
        label_index += 1