import os
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import torch
import pickle
import logging
from mpl_toolkits.axes_grid1 import ImageGrid

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rotmatrix(radian):
    R = np.array([
        [np.cos(radian), np.sin(-radian), 0],
        [np.sin(radian),  np.cos(radian), 0],
        [             0,               0, 1]
    ])
    return R

def check_data(image, uvmap):
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.axis('off')
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title("UVmap")
    plt.axis('off')
    plt.imshow(uvmap)
    plt.show()

def plot_CDE(datasets_name, datasets_losses):
    """_summary_

    Args:
        datasets_name (tuple): tuple of dataset names(string), should follow the order of datasets_losses
        datasets_losses (tuple): tuple of list, each list record the loss of every data in dataset.
    """
    # randomly set
    min_interval = 50 
    left = 0; right = 0;
    
    for idx, (name, losses) in enumerate(zip(datasets_name,datasets_losses)) :
        count, bins_count = np.histogram(losses, bins=1000)
        if min_interval > bins_count[-1]-bins_count[0]:
            right = bins_count[-1]
            left = bins_count[0]
            min_interval = right - left
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        plt.plot(bins_count[1:], cdf, label=f"{name}({len(losses)}): m={np.mean(losses):.5f}, v={np.var(losses):.8f}")
    
    plt.xlim(left, right)
    plt.legend()
    plt.show()

def plot_PDE(datasets_name, datasets_losses):
    
    for name, losses in zip(datasets_name,datasets_losses) :
        count, bins_count = np.histogram(losses, bins=100)
        pdf = count / sum(count)
        plt.hist(bins_count[:-1], bins_count, weights=pdf, label=f"{name}: m={np.mean(losses):.5f}, v={np.var(losses):.8f}")
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
        
    plt.legend()
    plt.show()

def get_dataset_losses(D, criterion, dataset, path, device, num_data = -1, G=None):
    """To check the formulation of energy surface
    - Use the different dataset for Discriminator for reconstruction
    - Record error of each data in dataset, and export a log file of list

    Args:
        D (_type_): Discriminator, Auto-Encoder
        criterion (_type_): L1 loss
        dataset (torch dataset): image+UVmap or only image is fine
        path (_type_): output file path
        device (_type_): _description_
        num_data (int, optional): how many data in dataset you want to record. Defaults to -1.
        G (_type_, optional): If you want to use the generator output as resource. Defaults to None.

    Returns:
        losses: list of each data in a dataset
    """
    if not os.path.isfile(path):
        
        losses = []
        max_num_data = len(dataset) if num_data == -1 else min(num_data, len(dataset))
        
        with torch.no_grad():
            for idx, data in enumerate(dataset):
                if G is None:
                    if type(data) is tuple:
                        image, uvmap = data[0], data[1]
                        D_in = torch.cat((image,uvmap), dim=0) # shape (6, 256, 256) CHW
                    else:
                        # Just set for other dataset ... e.g. "CelebA"
                        D_in = torch.cat((data, data), dim=0) # shape (6,256,256)
                else:
                    image = data[0]
                    G_in = torch.unsqueeze(image, dim=0).to(device)
                    G_out = G(G_in)
                    D_in = torch.cat((G_in,G_out), dim=1)
                    D_in = torch.squeeze(D_in)
                    
                
                D_in = torch.unsqueeze(D_in, dim=0) # shape (1, 6, 256, 256) NCHW
                D_in = D_in.to(device)
                D_out = D(D_in)
                
                # Just set for other dataset ... e.g. "CelebA"
                loss = criterion(D_out, D_in)
                
                # progress
                done = int(idx*100/max_num_data)
                undone = 100-done
                percent = float(idx/max_num_data*100)
                print('\r' + '[Progress]:[%s%s]%.2f%%;' % ('█'*done, ' '*undone, percent),
                end='')
                
                losses.append(loss.item())
                
                if idx == max_num_data-1:
                    break
                
        
        print()
        
        with open(path, "wb") as fp:
            pickle.dump(losses, fp)
    
    else:
        with open(path, "rb") as fp:
            losses = pickle.load(fp)
    
    logging.info(f"{path} is stored!!")
            
    return losses

def load_newest_ckpt(ckpt_dir, device):
    CKPT_paths = os.listdir(ckpt_dir) # list of "file" name
    if len(CKPT_paths) == 0:
        raise Exception("No ckpt in specified directory, 1. check dir path, 2. no trained model found")
    CKPT_paths = [ckpt_dir + x for x in CKPT_paths] # list of path+file name, cuz next line of code "getmtime()"
    CKPT_paths.sort(key = os.path.getmtime)
    ckpt = torch.load(CKPT_paths[-1], map_location=device) # laod the newest ckpt file
    
    logging.info(f"{CKPT_paths[-1]} is loaded")
    
    return ckpt

def get_landmarks(pos, uv_kpt_ind):
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        kpt: 68 3D landmarks. shape = (68, 3).
    '''
    kpt = pos[uv_kpt_ind[1,:], uv_kpt_ind[0,:], :]
    return kpt

def plot_landmarks_edge(kpts, ax=None):
    """
    Args:
        kpts (ndarray): shape should be (68, 2or3), needs at least the information of x and y coordinates
    """
    if ax is None:
        plt.plot(kpts[00:17,0], kpts[00:17,1]) # jaw
        plt.plot(kpts[17:22,0], kpts[17:22,1]) # right eyebraw
        plt.plot(kpts[22:27,0], kpts[22:27,1]) # left eyebraw
        plt.plot(kpts[27:31,0], kpts[27:31,1]) # nose bridge
        plt.plot(kpts[31:36,0], kpts[31:36,1]) # nose hole
        plt.plot(np.append(kpts[36:42,0], kpts[36,0]), np.append(kpts[36:42,1], kpts[36,1])) # right eye
        plt.plot(np.append(kpts[42:48,0], kpts[42,0]), np.append(kpts[42:48,1], kpts[42,1])) # left eye
        plt.plot(np.append(kpts[48:67,0], kpts[48,0]), np.append(kpts[48:67,1], kpts[48,1])) # lips
    else:
        ax.plot(kpts[00:17,0], kpts[00:17,1]) # jaw
        ax.plot(kpts[17:22,0], kpts[17:22,1]) # right eyebraw
        ax.plot(kpts[22:27,0], kpts[22:27,1]) # left eyebraw
        ax.plot(kpts[27:31,0], kpts[27:31,1]) # nose bridge
        ax.plot(kpts[31:36,0], kpts[31:36,1]) # nose hole
        ax.plot(np.append(kpts[36:42,0], kpts[36,0]), np.append(kpts[36:42,1], kpts[36,1])) # right eye
        ax.plot(np.append(kpts[42:48,0], kpts[42,0]), np.append(kpts[42:48,1], kpts[42,1])) # left eye
        ax.plot(np.append(kpts[48:67,0], kpts[48,0]), np.append(kpts[48:67,1], kpts[48,1])) # lips

def grid_of_alignment(G_in,pos_map,uv_kpt_ind,file_name,
                      figsize=(15., 20.),
                      nrows_ncols=(8, 6),
                      angle_list = None,
                      error_list = None,
                      name_list = None,
                      bm_version = "ori"
                      ):
    
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 
                     111, 
                     nrows_ncols=nrows_ncols,  # creates 2x2 grid of axes
                     axes_pad=0.01,  # pad between axes
    )
    
    for idx,ax in enumerate(grid):
        # plt image
        ax.imshow(G_in[idx].astype(np.int32))
        ax.get_yaxis().set_ticks([]); ax.get_xaxis().set_ticks([])

        # plt alignment
        kpt = get_landmarks(pos_map[idx], uv_kpt_ind) if bm_version == "ori" else pos_map[idx]
        ax.scatter(kpt[:,0], kpt[:,1], s=1, c="r", label="GAN_output")
        plot_landmarks_edge(kpt, ax)
        
        # show angle
        if angle_list is not None:
            if type(angle_list[idx]) is tuple: 
                ax.text(0,255.5,f"p:{int(angle_list[idx][0])}, y:{int(angle_list[idx][1])}, r:{int(angle_list[idx][2])}", color='red', fontweight="bold")
            else:
                ax.text(0,255.5,f"r:{int(angle_list[idx])}", color='red', fontweight="bold")
            
        if error_list is not None:
            ax.text(0,240.5,f"NME:{error_list[idx] * 100. :.2f} %", color='red', fontweight="bold")
        
        if name_list is not None:
            ax.text(0,225.5,f"{name_list[idx]}", color='red', fontweight="bold")
        
    plt.savefig(file_name)
    logging.info(f"{file_name} is stored!")
    
# def Error_functions(function_name):
#     function_list = ["NME_2D_68", "NME_3D_68", "NME_3D_face", "NME_2D_face"]
#     assert(function_name in function_list)
    
#     def error_func(G_GT, G_out, bbox):
        
#     return error_func
#     pass

def get_align68_losses(G, benchmark, inverse_transform, uv_kpt_ind, parent_dir, bm_version="ori"):
    
    if not (os.path.isfile(os.path.join(parent_dir, "NME_3D_68.log")) \
       and  os.path.isfile(os.path.join(parent_dir, "NME_2D_68.log")) \
       and  os.path.isfile(os.path.join(parent_dir, "Name.log"))):
        
        NME_3D_68_list = []
        NME_2D_68_list = []
        Name_list = []
        
        with torch.no_grad():
            #! remember, in ndarray, we are used to (H, W, C)
            # face_mask_np = io.imread('./uv_face_mask.png') / 255. # shape (256, 256)
            # face_mask_np = np.expand_dims(face_mask_np, axis=-1) # shape (256, 256, 1)
            # face_mask_mean_fix_rate = (256 * 256) / np.sum(face_mask_np) # due to we only care about face part in UV-MAP, but the data include the neck part

            for idx, (face_img, gt_pos) in enumerate(benchmark):
                
                G_in = torch.unsqueeze(face_img, dim=0)
                
                G_out = G(G_in)
                G_out = inverse_transform(G_out)
                G_out = np.squeeze(G_out)
                G_out_68 = get_landmarks(G_out, uv_kpt_ind) # shape: (68,3)
                
                if bm_version == 'ori':
                    G_GT = torch.unsqueeze(gt_pos, dim=0)
                    G_GT = inverse_transform(G_GT)
                    G_GT = np.squeeze(G_GT)
                    G_GT_68  = get_landmarks(G_GT, uv_kpt_ind) # shape: (68,3)
                
                else :
                    G_GT = gt_pos #! naming confused. because shape of "gt_pos" from reannotate version of benchmark is (2, 68)
                    G_GT_68 = np.zeros((68,3), dtype=G_GT.dtype)
                    G_GT_68[:,:2] = G_GT.T
                
                
                
                # z process
                G_out_68[:, 2] = G_out_68[:, 2] - G_out_68[:, 2].mean()
                G_GT_68[:, 2] = G_GT_68[:, 2] - G_GT_68[:, 2].mean()
                
                # L2_loss_3D_face = np.mean(np.sqrt(np.sum(np.square((G_out - G_GT)*face_mask_np), axis=-1))) * face_mask_mean_fix_rate
                # L2_loss_2D_face = np.mean(np.sqrt(np.sum(np.square((G_out[:,:,:2] - G_GT[:,:,:2])*face_mask_np), axis=-1))) * face_mask_mean_fix_rate
                L2_loss_3D_68 = np.mean(np.sqrt(np.sum(np.square(G_out_68 - G_GT_68), axis=1)))
                L2_loss_2D_68 = np.mean(np.sqrt(np.sum(np.square(G_out_68[:,:2] - G_GT_68[:,:2]), axis=1)))
                
                name, bbox, kpt = benchmark.__get_name_bbox_kpt__(idx)
                normalization_factor = np.sqrt(np.prod(bbox[1] - bbox[0])) # sqrt of bbox area
                
                # NME_3D_face = L2_loss_3D_face/normalization_factor
                # NME_2D_face = L2_loss_2D_face/normalization_factor
                NME_3D_68 = L2_loss_3D_68/normalization_factor
                NME_2D_68 = L2_loss_2D_68/normalization_factor
                
                # NME_3D_face_list.append(NME_3D_face)
                # NME_2D_face_list.append(NME_2D_face)
                NME_3D_68_list.append(NME_3D_68)
                NME_2D_68_list.append(NME_2D_68)
                Name_list.append(name)
                
                # progress
                done = int(idx*100/len(benchmark))
                undone = 100-done
                percent = float(idx/len(benchmark)*100)
                print('\r' + '[Progress]:[%s%s]%.2f%%;' % ('█'*done, ' '*undone, percent),
                end='')
            
            print()
            
            # sorting
            big_list = list(zip(Name_list, NME_2D_68_list, NME_3D_68_list))
            big_list = sorted(big_list, key= lambda x: int(x[0][5:]))
            Name_list, NME_2D_68_list, NME_3D_68_list = zip(*big_list)
            
            # with open(os.path.join(parent_dir, "NME_3D_face.log"), "wb") as fp:
            #     pickle.dump(NME_3D_face_list, fp)
            # with open(os.path.join(parent_dir, "NME_2D_face.log"), "wb") as fp:
            #     pickle.dump(NME_2D_face_list, fp)
            with open(os.path.join(parent_dir, "NME_3D_68.log"), "wb") as fp:
                pickle.dump(NME_3D_68_list, fp)
            with open(os.path.join(parent_dir, "NME_2D_68.log"), "wb") as fp:
                pickle.dump(NME_2D_68_list, fp)
            with open(os.path.join(parent_dir, "Name.log"), "wb") as fp:
                pickle.dump(Name_list, fp)

    else:
        with open(os.path.join(parent_dir, "NME_3D_68.log"), "rb") as fp:
            NME_3D_68_list = pickle.load(fp)
        with open(os.path.join(parent_dir, "NME_2D_68.log"), "rb") as fp:   
            NME_2D_68_list = pickle.load(fp)
        with open(os.path.join(parent_dir, "Name.log"), "rb") as fp:
            Name_list = pickle.load(fp)
        
    return Name_list, NME_2D_68_list, NME_3D_68_list


# == Analysis ==
def angle_error_analysis(NME_2D_68_list, angle_list):
        
        ## == statistic by zy ==
        # AFLW2000_3D_pitch = np.array([1813, 126, 61])
        # AFLW2000_3D_yaw = np.array([1312, 383, 305])
        # AFLW2000_3D_roll = np.array([1745, 184, 71])
        
        ## gimbol lock
        # AFLW2000_3D_pitch = np.array([1814, 128,  58,])
        # AFLW2000_3D_yaw = np.array([1313, 383, 304,])
        # AFLW2000_3D_roll = np.array([1746, 185,  69,])
        
        ## gimbol lock + roll 0~60~120~180 <- I choose this
        # AFLW2000_3D_pitch = np.array([1814, 128,  58,])
        # AFLW2000_3D_yaw = np.array([1313, 383, 304,])
        # AFLW2000_3D_roll = np.array([1931, 60, 9,])
        
    dict = {
        "pitch": (0, [30, 60, 90] , [1814, 128,  58,]),
        "yaw"  : (1, [30, 60, 90] , [1313, 383, 304] ),
        'roll' : (2, [60, 120, 180], [1931, 60, 9,]   )
    }
        
    choose = dict.keys()
    
    # def cov(error_list, angle_list, dict, c):
    #     angle_list = angle_list[:,dict[c][0]]
        
    #     angle_list = np.where(np.abs(angle_list) >=  dict[c][1][2],  dict[c][1][2] * np.sign(angle_list), angle_list) 
    #     angle_list = np.abs(angle_list)
                                
    #     return ((error_list - np.mean(error_list)) * (angle_list - np.mean(angle_list))).sum()/(len(error_list))
    
    for c in choose:
        l = [0.,0.,0.]
        color_list = []
        
        for error,angle in zip(NME_2D_68_list, angle_list):
            if np.abs(angle[dict[c][0]]) < dict[c][1][0]:
                color_list.append("green")
                l[0] += error
            elif np.abs(angle[dict[c][0]]) < dict[c][1][1]:
                color_list.append("blue")
                l[1] += error
            else:
                color_list.append("red")
                l[2] += error
        
        print(f"=={c}==")
        print(f"{0:03d}~{dict[c][1][0]:03d}:{l[0]/dict[c][2][0]*100:.2f}%")
        print(f"{dict[c][1][0]:03d}~{dict[c][1][1]:03d}:{l[1]/dict[c][2][1]*100:.2f}%")
        print(f"{dict[c][1][1]:03d}~{dict[c][1][2]:03d}:{l[2]/dict[c][2][2]*100:.2f}%")
        
        print(f"Mean: {(l[0]/dict[c][2][0] + l[1]/dict[c][2][1] + l[2]/dict[c][2][2])*100/3:.2f}%")
        print(f"Total Mean: {(l[0] + l[1] + l[2])/2000*100:.2f}%")
        
        print(f"max: {np.max(angle_list[:,dict[c][0]])}, min: {np.min(angle_list[:,dict[c][0]])}")
        
        
        # print(f"Covariance of {c} and error is: {cov(NME_2D_68_list, angle_list, dict, c)}")
        
        # fig = plt.figure(figsize=(50,20))
        # plt.scatter(angle_list[:,dict[c][0]],NME_2D_68_list)
        # plt.title(c, fontsize=40)
        # plt.savefig(f"../Image/img_npy_6/angle_{c}_with_error.png")
        
        
        plt.figure(figsize=(50,20))
    
        plt.bar(list(range(2000)),NME_2D_68_list, color=color_list)
        plt.title(c, fontsize=40)
    
        colors = {f'0~{dict[c][1][0]}':'green', f'{dict[c][1][0]}~{dict[c][1][1]}':'blue', f'{dict[c][1][1]}~{dict[c][1][2]}':'red'}         
        labels = list(colors.keys())
        handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
        plt.legend(handles, labels, fontsize=40)
    
        plt.savefig(f"../Image/img_npy_6/NME_2D_68_{c}.png")
        logging.info(f"../Image/img_npy_6/NME_2D_68_{c}.png saved!")
        print("="*20)