import cv2
import numpy as np
import math

import utils

def get_ROI():
    return [300, 100, 250, 250]


def main():
    key_ = ''
    classifier = utils.load_asl_classifier_v3(path='model/aslv2.h5', verbose=True)
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    im_height, im_width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (im_width,im_height))
    skipper = 0
    while ret:
        x, y, w,h = get_ROI()
        cv2.rectangle(frame, (x, y), (x+w, y+w), (255, 0, 0), 2)
        if skipper%10 == 0:
            cv2.imshow('roi', frame[y: y+h, x: x+w])
            roi = utils.preprocess_image_roi(frame[y: y+h, x: x+w])
            key_ = utils.classify_asl(classifier, roi)
        cv2.putText(frame, str(key_), (20, 20), 1, 2, (0, 0, 255))
        cv2.imshow('f', frame)
        out.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        # pre_roi = frame[y: y+h, x: x+w]
        ret, frame = cap.read()
        skipper += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()