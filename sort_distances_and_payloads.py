# Sort distances and payload at the same time based on shortest distance
# Replicating/Testing lines 134 - 177 of mtre4800-kawasaki-project\raspberry_pi_backup\programs_that_were_running_on_the_pi\segmentation2_kinect.py

class Payload():
    def __init__(self):
        self.x = 0
        # self.y = 0
        # self.z = 0
        # self.r = 0
        # self.type = 4
        self.distance = 100001
        # self.bounds = None
        self.selected = 0

dict = {}
sorted_dict = {}
new_distance = [10,7,17,5]

old_distance = [100,10,7,7,5]

# payloads = ['b1','w1','o1','b2']

for i in range(0,4):
    new_payload = Payload()
    # new_payload = payloads[i]
    new_payload.distance = new_distance[i]
    # payload_distance = new_distance[i]

    # Distance : Payload
    dict[new_payload.distance] = new_payload
    # dict[payload_distance] = payloads[i]

    # payloads.append(new_payload)
    # new_distance_list.append(new_payload.distance)
    # print(i)

print(str(dict))

# Reverse sorted dict since popitem() pops from back
for key in sorted(dict, reverse=True):
    sorted_dict[key] = dict[key]

print(sorted_dict)

for i in range(0, 7):
    if len(sorted_dict) != 0:
        distance_and_payload = sorted_dict.popitem()

        print(distance_and_payload[0], distance_and_payload[1])
    else:
        print("No more payloads.")

print(sorted_dict.items())
