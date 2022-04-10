import messaging
import socket
from time import time as now
from copy import deepcopy

# Organizes scada, cv, and ml programs
# Server in the Pi
'''
Starts 8090 port
Server has 1 static port
'''
class Tags:
    def __init__(self):
        self.vision_tags = { 'x':0 }
        self.robot_tags  = { 'x':0 }
        self.scada_tags  = { 'x':0 }
        self.lag = { 'vision':now(), 'robot':now(), 'scada':now() }
    
    def tags(self):
        return deepcopy( self.__dict__ )

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tag_server:
        tags = Tags()
        tag_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tag_server.bind(('', 8090))
        tag_server.listen()
        while True:
            connection, _ = tag_server.accept()
            with connection:
                try:
                    message = messaging.recv_msg(connection)
                except:
                    continue

                if message[0] == 'vision':
                    tags.vision_tags = message[1]
                elif message[0] == 'robot':
                    tags.robot_tags = message[1]
                elif message[0] == 'scada':
                    tags.scada_tags = message[1]

                tags.lag[message[0]] = now()
                messaging.send_message(tags.tags(), connection)

if __name__ == '__main__':
    main()