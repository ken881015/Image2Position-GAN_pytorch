import numpy as np
import matplotlib.pyplot as plt
import imageio

def get_landmarks(pos, uv_kpt_ind):
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        kpt: 68 3D landmarks. shape = (68, 3).
    '''
    kpt = pos[uv_kpt_ind[1,:], uv_kpt_ind[0,:], :]
    return kpt

def plot_landmarks_edge(kpts):
    """
    Args:
        kpts (ndarray): shape should be (68, 2or3), needs at least the information of x and y coordinates
    """
    
    plt.plot(kpts[00:17,0], kpts[00:17,1]) # jaw
    plt.plot(kpts[17:22,0], kpts[17:22,1]) # right eyebraw
    plt.plot(kpts[22:27,0], kpts[22:27,1]) # left eyebraw
    plt.plot(kpts[27:31,0], kpts[27:31,1]) # nose bridge
    plt.plot(kpts[31:36,0], kpts[31:36,1]) # nose hole
    plt.plot(np.append(kpts[36:42,0], kpts[36,0]), np.append(kpts[36:42,1], kpts[36,1])) # right eye
    plt.plot(np.append(kpts[42:48,0], kpts[42,0]), np.append(kpts[42:48,1], kpts[42,1])) # left eye
    plt.plot(kpts[48:67,0], kpts[48:67,1]) # lips

# File io
img = plt.imread("./face/AFW_261068_1_0.jpg")
npy = np.load("./uvmap/AFW_261068_1_0.npy")
jpg = plt.imread("./uvmap/AFW_261068_1_0_posmap.jpg", format=np.float32)
angle = np.load("./uvmap/AFW_261068_1_0_angle.npy")
uv_kpt_ind = np.loadtxt('./uv_kpt_ind.txt').astype(np.int32) # 2 x 68 get kpt


jpg_kpt = get_landmarks(jpg, uv_kpt_ind)
npy_kpt = get_landmarks(npy, uv_kpt_ind)

fig = plt.figure()

# ax = fig.add_subplot(1,2,1,projection="3d")
# ax.scatter3D(uvmap[:,0], uvmap[:,1], uvmap[:,2], color=color)
# ax.set_xlim(0,256);ax.set_ylim(0,256);ax.set_zlim(0,256);
# ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

# ax = fig.add_subplot(1,2,2,projection="3d")
# ax.scatter3D(uv_kpt[:,0], uv_kpt[:,1], uv_kpt[:,2], color=color_kpt)
# ax.set_xlim(0,256);ax.set_ylim(0,256);ax.set_zlim(0,256);
# ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

plt.imshow(img)
plt.scatter(jpg_kpt[:,0], jpg_kpt[:,1], s=5, c="r", label="jpg")
# plt.scatter(npy_kpt[:,0], npy_kpt[:,1], s=5, c="b", label="npy")

plot_landmarks_edge(jpg_kpt)

plt.legend()

plt.savefig("./alignment_68/align_by_jpg.png")