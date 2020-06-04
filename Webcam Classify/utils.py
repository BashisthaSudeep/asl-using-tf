import tensorflow as tf
import numpy as np
import os
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
# import tensorflow.keras
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Conv2D, Dropout

from tensorflow.keras.models import Model
import tensorflow.keras as keras

_score_thresh = 0.27

MODEL_NAME = 'hand_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')

# char_to_num = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18, 'j': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 23, 'o': 24, 'p': 25, 'q': 26, 'r': 27, 's': 28, 't': 29, 'u': 30, 'v': 31, 'w': 32, 'x': 33, 'y': 34, 'z': 35}
char_to_num = {chr(65 + i): i for i in range(26)}
char_to_num['_'] = 26
char_to_num['delete'] = 27
char_to_num['insert'] = 28
nb_classes = 29

def load_inference_graph():

    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess

# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


def load_classifier(path='asl3.h5'):
    backbone = VGG16(input_shape=(400, 400, 3), include_top=False)

    model = Sequential()
    model.add(backbone)
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # model = keras.models.load_model("asl.h5")
    model.load_weights(path)

    return model

def classify(model, roi):

    roi = cv2.resize(roi, (400, 400))
    # cv2.imwrite('out.jpg', roi)   
    roi = roi.reshape((1,) + roi.shape)
    roi = roi.astype(float)/255.0
    predection = model.predict_classes(roi)
    # Printing the predicted class name.
    key = (key for key, value in char_to_num.items() if value == predection[0]).__next__()
    return key


def get_roi(img1, img2):
    diff = cv2.absdiff(img1, img2)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # th = 30
    # imask =  mask>th

    # canvas = np.zeros_like(img2, np.uint8)
    # canvas[imask] = img2[imask]

    # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    blur = cv2.bilateralFilter(mask,9,75,75)
    ret, thresh = cv2.threshold(blur,50, 245, 0)
    # _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # if len(contours) == 0:
    #     return
    # cv2.drawContours(img2.copy(), [contours[0]], 0, (255, 0, 0), 3)
    out = cv2.bitwise_and(img2, img2, mask=thresh)
    cv2.imshow('roi', out)
    return cv2.resize(out, (400,400))

def load_asl_classifier(path='als_alpha.h5'):
    n_classes = 29
    target_dims = (100, 100, 3)
    my_model = Sequential()
    my_model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=target_dims))
    my_model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
    my_model.add(Dropout(0.5))
    my_model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
    my_model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
    my_model.add(Dropout(0.5))
    my_model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))
    my_model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
    my_model.add(Flatten())
    my_model.add(Dropout(0.5))
    my_model.add(Dense(512, activation='relu'))
    my_model.add(Dense(n_classes, activation='softmax'))

    # model = keras.models.load_model("asl.h5")
    my_model.load_weights(path)

    return my_model

def load_asl_classifier_v2(path='asl_pre.h5'):
    pretrained_model = VGG16(include_top=False, input_shape=(100, 100, 3))
    x = pretrained_model.output
    x = Flatten()(x)
    predictions = Dense(29, activation='softmax')(x)
    model = Model(inputs=pretrained_model.input, outputs=predictions)
    # Train top layer
    for layer in pretrained_model.layers:
        layer.trainable = False
    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', 
                    optimizer=optimizer, 
                    metrics=['accuracy'])
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
    
    model.load_weights('asl_pre.h5')

    return model

def load_asl_classifier_v3(path='asl.h5', verbose=False):
    print('------------> reading model from', path)
    pretrained_model = InceptionV3(include_top=False, input_shape=(299,299,3), weights="imagenet")

    # pretrained_model.trainable = False

    x = pretrained_model.output
    x = Flatten()(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dropout(0.5)(x)
    predictions = Dense(29, activation='softmax')(x)
    model = Model(inputs=pretrained_model.input, outputs=predictions)
    if verbose:
        print(model.summary())
        print(model.__dict__.keys())
        print(model.inputs)
        # print('----->', model.get_layer('input_3 (InputLayer)'))
    return model

def preprocess_image_roi(roi):
    roi = cv2.resize(roi, (299, 299))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = roi.astype(np.float32)
    roi -= roi.mean()
    # Apply featurewise_std_normalization to test-data with statistics from train data
    roi /= (roi.std() + keras.backend.epsilon())

    # roi = roi.reshape((1,) + roi.shape)
    roi = np.expand_dims(roi, axis=0)

    return roi

def classify_asl(model, roi):
    try:
        predection = model.predict_classes(roi)
    except:
        result = model.predict(roi)
        predection = [result.argmax()]
    # Printing the predicted class name.
    key = (key for key, value in char_to_num.items() if value == predection[0]).__next__()
    return key