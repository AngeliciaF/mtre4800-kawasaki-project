from pymodbus.client.sync import ModbusTcpClient as ModbusClient
 
class Robot:
   __payload_presence = 0
   __select_conveyor = 1
   __photo_eye = 2
 
   __home = 0
   __gripper_away = 1
   __open = 2
 
   __lid_placed = 0
   __lid_request = 0
 
   def __init__(self, ip):
       self.ip_address = ip
       self.x = 0 
       self.y = 0
       self.z = 0
       self.r = 0
       self.gripper_away = False
       self.home = False
       self.open = True
       self.lid_placed = False
       self.input_signals = 0
       self.output_signals = 0
 
   def tags(self):
       tag_set = { 'home':self.home, 'gripper_away':self.gripper_away, 'open':self.open, 'lid_placed':self.lid_placed }
       return tag_set
 
   def payload_presence(self, level):
       self.output_signal(self.__payload_presence, level)
  
   def select_conveyor(self, level):
       self.output_signal(self.__select_conveyor, level)
 
   def gripper_eye(self, level):
       self.output_signal(self.__photo_eye, level)
 
   def lid_request(self, level):
       self.output_signal(self.__lid_request, level)
 
   #from robot
   def input_signal(self, signal):
       return self.get_bit(self.input_signals, signal)
 
   #to robot
   def output_signal(self, signal, level):
       self.output_signals = self.write_bit(self.output_signals, signal, level)
 
   def set_payload(self, payload):
       self.x = payload.x
       self.y = payload.y
       self.z = payload.z
       self.r = payload.r
 
   def get_bit(self, word, signal):
       brep = bin(word)
       if len(brep) <= signal+2:
           return 0
       return int(bin(word)[-(signal+1)])
 
   def set_bit(self, word, signal):
       return word | (1<<signal)
 
   def clear_bit(self, word, signal):
       return word & ~(1<<signal)
 
   def write_bit(self, word, signal, level):
       if level:
           return self.set_bit(word, signal)
       else:
           return self.clear_bit(word, signal)
 
   def update(self):
       gX = self.x + 10000
       gY = self.y + 10000
       gZ = self.z + 10000
       gR = self.r + 10000
       try:
           with ModbusClient(self.ip_address) as client:
               result1 = client.write_register(1024, self.output_signals, unit=0x1)
               #print('output:', self.output_signals, result1)
               result2 = client.write_register(1025, int(gX), unit=0x1)
               result3 = client.write_register(1026, int(gY), unit=0x1)
               result4 = client.write_register(1027, int(gZ), unit=0x1)
               result5 = client.write_register(1028, int(gR), unit=0x1)
 
               output = client.read_holding_registers(    0, 4, unit=0x1 )
               input  = client.read_holding_registers( 1024, 5, unit=0x1 )
               #print(output, input)
               #print( self.x, self.y, self.z, self.r )
               self.input_signals = output.registers[0]
               self.gripper_away = self.get_bit(self.input_signals, self.__gripper_away)
               self.home = self.get_bit(self.input_signals, self.__home)
               self.open = self.get_bit(self.input_signals, self.__open)
               self.lid_placed = self.get_bit(self.input_signals, self.__lid_placed)
           return True
       except Exception as e:
           #print(e)
           return False
 
def main():
    from payload import Payload
    import messaging
    from time import time as now
    from time import sleep

    # TODO: Make this a Node_RED node
    left = Robot('192.168.0.14')

    # Commented out the coms to the right robot
    # since the other team will be handling this
    # right = Robot('192.168.0.13')
    cycle_time = 1/30 # how often to communicate

    payload = Payload()
    # TODO: Maybe change for Node_RED node
    left.payload_presence((payload.selected==1))
    left.select_conveyor((payload.type==0))
    # right.lid_request(False)

    bg = now()
 
    while True:
        start_time = now()
 
        left.set_payload(payload)
        left.update()
 
        # right.update()
 
        tag_set = left.tags()
        tag_set.pop('lid_placed')
        # rtags = right.tags()
        # rtags.pop('home')
        # rtags.pop('gripper_away')
        # rtags.pop('open')
        # tag_set.update(rtags)
      
        #{ 'home':self.home, 'gripper_away':self.gripper_away, 'open':self.open, 'lid_placed':self.lid_placed }
 
        vision = messaging.client_send('robot', tag_set, True)
        if vision is not None and 'y' in vision['vision_tags'].keys():
            #print(vision['scada_tags'][''])
            payload.x = vision['vision_tags']['x']
            payload.y = vision['vision_tags']['y']
            payload.z = vision['vision_tags']['z']
            payload.r = vision['vision_tags']['r']
            
            # TODO: Maybe change for Node_RED node
            left.payload_presence((vision['vision_tags']['number_of_payloads'] > 0))
            left.select_conveyor((vision['vision_tags']['type'] == 0))
            left.gripper_eye(vision['scada_tags']['photo_eye'])
            #print(vision['scada_tags']['lid_requested'])
            # right.lid_request(vision['scada_tags']['lid_requested'])
        else:
            # TODO: Maybe change for Node_RED node
            left.output_signals = 0
 
        if now() - start_time < cycle_time:
            sleep(now()-start_time)
 
if __name__ == '__main__':
   main()
