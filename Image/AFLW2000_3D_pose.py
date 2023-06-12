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

# gimbal lock process
for name, angle in zip(name_list, angle_list):
    # pitch
    if np.abs(angle[0]) > 270:
        # print("pitch")
        # print("ori:", name, angle)
        angle[0] -= 360 * np.sign(angle[0])
        # print("upt:", name, angle)
    
    #yaw
    if np.abs(angle[1]) > 270:
        # print("yaw")
        # print("ori:", name, angle)
        angle[1] -= 360 * np.sign(angle[1])
        # print("upt:", name, angle)
    
    #roll
    if np.abs(angle[2]) > 180:
        # print("roll")
        # print("ori:", name, angle)
        angle[2] -= 360 * np.sign(angle[2])
        # print("upt:", name, angle)
    
    # if np.abs(angle[1]) < 110 and np.abs(angle[1]) > 70:
    #     print(name, angle)



# sorted by file name idx
fname_with_angle = list(zip(name_list, angle_list))
fname_with_angle = sorted(fname_with_angle, key = lambda x: int(x[0][5:]))

cnt = 0

for i in range(2000):
    if np.abs(angle_list[i,2]) < 165:
        cnt += 1

print(cnt)

# with open("../Main/AFLW2000-3D_fname_pose_gimbal_lock", "wb") as fp:
#     pickle.dump(fname_with_angle, fp)







