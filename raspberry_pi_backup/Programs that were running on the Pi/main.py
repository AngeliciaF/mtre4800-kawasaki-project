import cv2
from AImodel import AIModel
import cameraInterface
from payload import Payload
import segmentation2 as segmentation
from time import time as now
from time import sleep
import messaging
import numpy
import cameraInterface
import videoServer as flask
import threading
import multiprocessing
import math
from copy import deepcopy
from math import sqrt
'''
x = numpy.zeros((3,15,20))
x1 = numpy.reshape(x[0], (1,300))
x2 = numpy.reshape(x[1], (1,300))
x3 = numpy.reshape(x[2], (1,300))
y = numpy.concatenate((x1, x2, x3), 1)
print(x.shape, y.shape)
exit()
'''
threading.Thread(target=lambda: flask.main()).start()

#model = AIModel((1296,976), 440)
model = AIModel()
heights = [-254, -374, -343, -275, 0 ]

template = cv2.imread('floor.Bmp')
template = cv2.resize(template,(648,488))
template = template[60:480,70:610]
template = cv2.resize(template, (160,120))

scada = { 'robot_tags':{'home':True} }
prediction = 4

cam = cameraInterface.Camera()
for frame in cam.next():
    start = now()
    img = cam.prepare(frame)

    payloads = segmentation.getPayloads(img)
    
    bounding_box = None
    x = 0
    y = 0
    selected = 0
    for index, payload in enumerate(payloads):
        if payload.selected:
            selected = index
            bounding_box, sample = segmentation.getPayload(payload, img)
            if scada['robot_tags']['home']:
                prediction = model.getPrediction(sample, payload)
                payload.z = heights[payload.type]
            else:
                payload.type = prediction
                payload.z = heights[payload.type]
            x, y = segmentation.convert_units(payload.x, payload.y, img.shape)

    if scada['robot_tags']['home']:
        segmentation.draw_payloads(img, payloads, bounding_box, model.labels)

    tag_set = Payload().tags()
    if len(payloads) > 0:
        payloads[selected].x = x
        payloads[selected].y = y
        cv2.putText(img, "X: " + str(round(payloads[selected].x,0)) + 'mm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(img, "Y: " + str(round(payloads[selected].y,0)) + 'mm', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        tag_set = deepcopy(payloads[selected].tags())
    tag_set['number_of_payloads'] = len(payloads)

    scada = messaging.client_send('vision', tag_set, True)
    #print(scada['scada_tags'])
    img = cv2.resize(img,(648,488))
    flask.push(img)
    #cv2.imshow("frame", img)
    #if (cv2.waitKey(1) & 0xFF) == 27:
    #    break

    #cycle = (1/30)-(now()-start)
    #if cycle > 0.001:
    #    sleep(cycle)

    #print(round(now()-start,3)*1000, 'mS')
#cam.shutdown()