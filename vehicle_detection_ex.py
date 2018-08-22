import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import time
from pylab import savefig

start = time.time()

# What model to use# What
#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_NAME = 'faster_rcnn_nas_lowproposals_coco_2018_01_28'  ############### 성능 90% 이상 이지만 모델의 용량이 커서 훈련 시간 상당히 오래 김
########3 최소 GTX 1060 이상 추천!!!!!!!!!!!!!!!!!!
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17' ############## 성능 80% 정도

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = MODEL_NAME + '/mscoco_label_map.pbtxt'
# PATH_TO_LABELS = MODEL_NAME + '/pascal_label_map.pbtxt'

NUM_CLASSES = 90
# NUM_CLASSES = 20

def load_detection_model(PATH_TO_CKPT=PATH_TO_CKPT):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def detection_operations():
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    return image_tensor, detection_boxes, detection_scores, detection_classes, num_detections

def process_image(image):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(image, np.squeeze(boxes), np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=8)
            return image

def clear_cache():
    process_image.cache = None

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = load_detection_model()
image_tensor, detection_boxes, detection_scores, detection_classes, num_detections = detection_operations()

write_output  = 'car_output.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first n seconds
# clip1 = VideoFileClip("car.mp4").subclip(0,3)
clip1 = VideoFileClip("car.mp4")
#clear_cache()
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(write_output, audio=False)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGES_PATHS = [os.path.join(PATH_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 4)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
print(TEST_IMAGES_PATHS)

for image_path in TEST_IMAGES_PATHS:
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    image_processed=process_image(image_np)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_processed)
    plt.show()

end = time.time()
training = end - start
print('훈련 시간 : ', training, '(초)')




