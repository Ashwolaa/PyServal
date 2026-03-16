#%%
import numpy as np
import os
import matplotlib.pyplot as plt

folder = r"/home/laboratory/Tpx3Cam/PyServal/data/"
filename = r"rec_20260311_102937_saver1_events.dat"
filename = r"acquisition_20260311_121202_saver0_events.dat"
file = os.path.join(folder, filename)

luna_process = r"Luna/bin/tpx3dump.exe"


#%%
data = np.fromfile(file, dtype=np.dtype([
    ("event_num", "<u8"),
    ("x", "<u2"),
    ("y", "<u2"),
    ("tof", "<f8"),
    ("tot", "<u4"),
]))
#%%
f,axs = plt.subplots(1, 3, figsize=(12, 6))
axs[0].hist(data["tof"], bins=100)
axs[1].hist(data["tot"], bins=100)
axs[2].hist2d(data["x"], data["y"], bins=100)
plt.show()
#%%
# f,axs = plt.subplots(1, 3, figsize=(12, 6))
plt.figure(figsize=(12, 6))
plt.hist2d(data["tof"], data["tot"], bins=100)
plt.show()


#%%

folder = r"/home/laboratory/Tpx3Cam/PyServal/data/"
filename = r"test.h5"
file = os.path.join(folder, filename)

import h5py


with h5py.File(file, "r") as f:
    print(f"Keys: {list(f.keys())}")
