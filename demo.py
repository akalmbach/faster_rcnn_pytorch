import cv2, os, time
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer

import matplotlib.pyplot as plt

def test():

    model_file = 'demo/VGGnet_fast_rcnn_iter_70000.h5'
    detector = FasterRCNN()
    network.load_net(model_file, detector)
    detector.cuda()
    detector.eval()
    print('load model successfully!')

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('demo', cv2.WINDOW_NORMAL)

    while True:
        # im_file = 'data/VOCdevkit2007/VOC2007/JPEGImages/009036.jpg'
        ret, image_webcacm = cap.read()
        image = cv2.resize(image_webcacm, (800, 600))

        cv2.imshow('demo', image)

        score, prob, boxes = detector.run_rpn(image)
        
        prob_tens = prob.squeeze().reshape((2,3,3,37,50))

        for scale in [0,1,2]:
            print(np.mean(prob_tens[1,:,scale]))

        prob_scale = np.sum(prob_tens, axis=1)

        f,axarr = plt.subplots(2,3)
        for pos in [0,1]:
            for i,ax in enumerate(axarr[pos]):
                ax.imshow(prob_scale[pos,i], vmin=0, vmax=3, cmap='bone')

        plt.show()

        # f,axarr = plt.subplots(4,3)
        # for i,ax in enumerate(axarr.flatten()):
        #     ax.imshow(box_tens[i].transpose([1,2,0]), cmap='bone')
            
        # plt.show()

        # dets, scores, classes = detector.detect(image, 0.3)

        # im2show = np.copy(image)
        # for i, det in enumerate(dets):
        #     det = tuple(int(x) for x in det)
        #     cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
        #     cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
        #                 1.0, (0, 0, 255), thickness=1)

        # cv2.imshow('demo', im2show)
        # cv2.waitKey(1)



if __name__ == '__main__':
    test()
