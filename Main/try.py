import numpy as np
import scipy.io as sio
import pickle
import os
import matplotlib.pylab as plt
import matplotlib.patches as patches

mat_list = os.listdir("/home/vlsilab/Dataset/AFLW2000_all/")

mat_list = [fname for fname in mat_list if fname[-3:] == 'mat']
name_list = [fname[:-4] for fname in mat_list]

radian_list = [np.squeeze(sio.loadmat("/home/vlsilab/Dataset/AFLW2000_all/"+fname)["Pose_Para"])[:3] for fname in mat_list]
angle_list = [np.rad2deg(radian) for radian in radian_list]
angle_list = np.abs(angle_list).astype(np.int16) % 360 * np.sign(angle_list)

fname_with_angle = list(zip(name_list, angle_list))

fname_with_angle = sorted(fname_with_angle, key = lambda x: int(x[0][5:]))

with open("../Main/AFLW2000-3D_fname_pose", "wb") as fp:
    pickle.dump(fname_with_angle, fp)







