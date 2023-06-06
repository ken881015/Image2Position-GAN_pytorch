import numpy as np
import scipy.io as sio
import os
import matplotlib.pylab as plt
import matplotlib.patches as patches

bchm_dir = "/home/vlsilab/Dataset/Img2Pos_test/AFLW2000_all-crop/"
img_dir_name = "image00180" #! Just need to change the image directory name!!
file_path = bchm_dir + os.path.join(img_dir_name,img_dir_name)


cropped_image = plt.imread(file_path+"_cropped.jpg")
cropped_npy = np.load(file_path+"_cropped_uv_posmap.npy")
bbox_info = sio.loadmat(file_path+"_bbox_info.mat")

print(np.max(cropped_npy), np.min(cropped_npy))

bbox = bbox_info["Bbox"]

width = bbox[1,0] - bbox[0,0]
height = bbox[1,1] - bbox[0,1]

fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(cropped_image)

# Create a Rectangle patch
rect = patches.Rectangle(bbox[0], width, height, linewidth=1, edgecolor='r', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

plt.savefig("face_with_bbox.jpg")






