from pymodbus.version import version
from pymodbus.server.asynchronous import StartTcpServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from twisted.internet.task import LoopingCall
import logging
import messaging

'''
1 = x
2 = y
3 = z
4 = r
5 = type
6 = gripper away
7 = no payloads
'''


# SCADA server
# Goes straight to SCADA from modbus

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.CRITICAL)

def updating_writer(a):
    context = a[0][0]
    func = 3

    address = 0
    values = context.getValues(func, address, count=32)

    conveyor_full = values[6]
    gripper_photo_eye = values[9]
    lid_requested = values[10]
    served =  {'conveyor_full':conveyor_full, 'photo_eye':gripper_photo_eye, 'lid_requested':lid_requested}
    #print(values)
    scada = messaging.client_send('scada',served, True)
    if 'y' in scada['vision_tags'].keys():
        values[0] = int(abs(scada['vision_tags']['x']+10000))
        values[1] = int(abs(scada['vision_tags']['y']+10000))
        values[2] = int(abs(scada['vision_tags']['r']+10000))
        values[3] = int(abs(scada['vision_tags']['type']))
        values[5] = int(abs(scada['vision_tags']['number_of_payloads']))
    if 'open' in scada['robot_tags'].keys():
        #print(scada['robot_tags'])
        values[4] = int(abs(scada['robot_tags']['gripper_away']))
        values[7] = int(abs(scada['robot_tags']['home']))
        values[8] = int(abs(scada['robot_tags']['open']))
        values[11] = int(abs(scada['robot_tags']['lid_placed']))

    context.setValues(func, address, values)

    printv = context.getValues(3, 0, count=10)
    #print("new values: " + str(printv))


def run_updating_server():
    store = ModbusSlaveContext(di=None, co=None, hr=ModbusSequentialDataBlock(0, [0]*32), ir=None)
    context = ModbusServerContext(slaves=store, single=True)
    
    identity = ModbusDeviceIdentification()
    identity.VendorName = 'pymodbus'
    identity.ProductCode = 'PM'
    identity.VendorUrl = 'http://github.com/riptideio/pymodbus/'
    identity.ProductName = 'pymodbus Server'
    identity.ModelName = 'pymodbus Server'
    identity.MajorMinorRevision = version.short()
    
    time = 0.1  # 5 seconds delay
    loop = LoopingCall(f=updating_writer, a=(context,))
    loop.start(time, now=False)
    StartTcpServer(context, identity=identity, address=("192.168.0.100", 5019))


if __name__ == "__main__":
    run_updating_server()