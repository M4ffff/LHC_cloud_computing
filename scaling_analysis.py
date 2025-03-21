import subprocess
import matplotlib.pyplot as plt
import time
import json
import numpy as np

# json 1 == range 1 to 3
# json 2 == range 1 to 4
# json 3 == run_time_speed set to 1


scaling_timings = {}
other_scaling_timings = {}

run_time_speeds = np.arange(0.2,1.05,0.2)
num_containers = np.arange(1,5)

scaling_timings_list = []
other_scaling_timings_list = []
    
for container in num_containers:

    start = time.time()
    output = subprocess.run(["docker", "compose", "up",  "--scale", f"consumer={container}", "--abort-on-container-exit"], text=True,  capture_output=True)
    # print(output)
    end = time.time()
    elapsed = end-start
    print(f"\nelapsed: {elapsed}\n")
    other_elapsed = str(output).split("data timing:")
    print(f"\n other elapsed: {other_elapsed[1][:7]}")
    
    scaling_timings_list.append( elapsed )
    other_scaling_timings_list.append( float(other_elapsed[1][:7]) )
    
    scaling_timings[container] = scaling_timings_list
    other_scaling_timings[container] = other_scaling_timings_list
        
        

with open("scaling_timings4.json", "w") as file: 
    json.dump(scaling_timings, file)

with open("other_scaling_timings4.json", "w") as file: 
    json.dump(other_scaling_timings, file)
    
    
    
with open("scale timings/scaling_timings4.json", "r") as file: 
    scaling_timings = json.load(file)
    
with open("scale timings/other_scaling_timings4.json", "r") as file: 
    other_scaling_timings = json.load(file)



fig,ax = plt.subplots(1,2,figsize=(16,6))

for i in (num_containers):
    ax[0].plot(np.arange(1,5), scaling_timings, label=f"{i} containers")
    ax[1].plot(np.arange(1,5), other_scaling_timings, label=f"{i} containers")
ax[0].set_xlabel("number of consumers")
ax[0].set_ylabel("time / s")
# ax[0].set_yscale("log")
ax[1].set_xlabel("number of consumers")
ax[1].set_ylabel("time / s")
# ax[0].set_yscale("log")
plt.show()


# fig,ax = plt.subplots(1,2, figsize=(10,5))
# ax[0].scatter(np.arange(1,5), scaling_timings)
# ax[1].scatter(np.arange(1,5), other_scaling_timings)
# ax[0].set_xlabel("number of consumers")
# ax[1].set_xlabel("number of consumers")
# ax[0].set_ylabel("time / s")
# ax[1].set_ylabel("time / s")
# plt.show()
