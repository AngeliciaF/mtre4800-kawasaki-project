import socket
import pickle
from time import time as now
import struct
import cv2
import numpy

def send_message(message, socket):
    data = pickle.dumps(message)
    data = struct.pack('>I', len(data)) + data
    socket.sendall(data)

def recv_msg(socket):
    # Read message length and unpack it into an integer
    packet = socket.recv(4)
    length = struct.unpack('>I', packet)[0]
    # Read the message data
    data = bytearray()
    while len(data) < length:
        packet = socket.recv(length - len(data))
        if not packet:
            return None
        data.extend(packet)
    message = pickle.loads(data)
    return message

# Linux controls sockets
# Tell it to reuse socket address
# Open client socket
# Send to server socket
# Recieve message
# Close client socket
def client_send(client, message, receive=False):
    try:
        # Use with to open and automatically close the socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as local:
            #local.settimeout(0.01)
            local.connect(('localhost', 8090))
            send_message( (client, message), local)
            if receive:
                return recv_msg(local)
    except:
        print("comms error")
        if receive:
            return None