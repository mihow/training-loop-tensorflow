# IMPORTS

import numpy as np
import os
import time
import tensorflow as tf
import json


def load_graph(model_file):
  # Load a (frozen) Tensorflow model into memory.
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


graph = load_graph('tf_files/retrained_graph.pb')
labels = load_labels('tf_files/retrained_labels.txt')


# Helper code
def load_image_into_numpy_array(image,
        input_height=224, input_width=224,
				input_mean=0, input_std=255):
    #image = np.array(image)[:, :, 0:3]
    float_caster = tf.cast(image, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result
    # (im_width, im_height) = image.size
    # return np.array(image.getdata()).reshape((input_height, input_width, 3)).astype(np.uint8)


def predict(image):
    t = load_image_into_numpy_array(image)

    input_name = "import/input"
    output_name = "import/final_result"
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
        end = time.time()
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]

    print("\nEvaluation time (1-image): {:.3f}s\n".format(end - start))
    template = "{} (score={:0.5f})"
    for i in top_k:
        print(template.format(labels[i], results[i]))
    best_guess = "%s %s" % (results[top_k[0]], labels[top_k[0]])
    return best_guess



def test():
  from PIL import Image
  import glob

  # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
  PATH_TO_TEST_IMAGES_DIR = 'tf_files/studio/redvase' #cwh
  TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR+'/*.jpg')

  for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      response = predict(image)
      print("returned JSON: \n%s" % response)

if __name__ == '__main__':
  test()