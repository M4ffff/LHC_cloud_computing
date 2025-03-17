import subprocess
import matplotlib.pyplot as plt
import time
import json
import numpy as np

# scaling_timings = []
# other_scaling_timings = []

# for i in range(1,5):
#     start = time.time()
#     ouput = subprocess.run(["docker", "compose", "up",  "--scale", f"consumer={i}", "--abort-on-container-exit"], text=True,  capture_output=True)
#     # print(ouput)
#     end = time.time()
#     elapsed = end-start
#     print(f"\nelapsed: {elapsed}\n")
#     other_elapsed = str(ouput).split("data timing:")
#     print(f"\n other elapsed: {other_elapsed[1][:7]}")
    
#     scaling_timings.append( elapsed )
#     other_scaling_timings.append( float(other_elapsed[1][:7]) )

# with open("scaling_timings2.json", "w") as file: 
#     json.dump(scaling_timings, file)

# with open("other_scaling_timings2.json", "w") as file: 
#     json.dump(other_scaling_timings, file)
    
    
    
with open("scaling_timings2.json", "r") as file: 
    scaling_timings = json.load(file)
    
with open("other_scaling_timings2.json", "r") as file: 
    other_scaling_timings = json.load(file)



fig,ax = plt.subplots(1,2, figsize=(10,5))
ax[0].scatter(np.arange(1,5), scaling_timings)
ax[1].scatter(np.arange(1,5), other_scaling_timings)
ax[0].set_xlabel("number of consumers")
ax[1].set_xlabel("number of consumers")
ax[0].set_ylabel("time/s")
ax[1].set_ylabel("time/s")
plt.show()
