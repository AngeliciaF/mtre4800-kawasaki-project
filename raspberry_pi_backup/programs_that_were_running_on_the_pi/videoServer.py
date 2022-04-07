from flask import Flask, Response
from flask_sockets import Sockets
import json
import queue
import cv2

app = Flask(__name__)
sockets = Sockets(app)

imq = queue.Queue()

@app.route('/')
def hello():
    return 'Hello World!'

def gen(q):
    while True:
        image = q.get()
        frame = cv2.imencode('.jpg', image)[1]
        
        yield ( b'--frame\r\n'
                b'Content-Type:image/jpeg\r\n'
                b'Content-Length: ' + f"{len(frame)}".encode() + b'\r\n'
                b'\r\n' + frame.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(imq), mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()

def push_image(obj):
    imq.put(obj)
