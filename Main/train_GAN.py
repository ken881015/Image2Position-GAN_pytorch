import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
import subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import argparse
import logging
import json
import time
import pickle
import matplotlib.pyplot as plt
import skimage.io as io
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from Model.Generator import *
from Model.Discriminator import *
from Data_process import *
from Utils import *


# parser configuration
parser = argparse.ArgumentParser()

# ==file io==
parser.add_argument("--input_dir", required=True, help="path to folder containing training data")
parser.add_argument("--bchm_dir", required=True, help="path to folder containing benchmark data")

parser.add_argument("--ckpt_dir", required=True, help="where to put ckpts")
parser.add_argument("--img_dir", required=True, help="path for storing generated images")
parser.add_argument("--tblog_dir", required=True, help="path to log file")


# ==model & training configuration==
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("-e","--epochs", type=int, help="number of training epochs")
# parser.add_argument("--seperable_conv", action="store_true", help="Imply seperable conv layer to minify model and calculation")

# hyperparameter
parser.add_argument("--lr", type=float, default=0.00005, help="initial learning rate for adam")
parser.add_argument("--decay_step", type=int, default=30, help="Step(unit:epoch) to decay learning rate")
parser.add_argument("--beta1", type=float, default=0.9 , help="Adam's Hyper parameter")
parser.add_argument("--beta2", type=float, default=0.99, help="Adam's Hyper parameter")
parser.add_argument("--l1_weight", type=float, default=1, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1, help="weight on GAN term for generator gradient")
parser.add_argument("--batch_size", type=int, default=16, help="number of images in batch")

# from BEGAN
parser.add_argument('--gamma', default=0.5, type=float, help='equilibrium balance ratio; diversity ratio proportional to the diversity of generated images')
parser.add_argument('--lambda_k', default=0.001, type=float, help='the proportional gain of k')

# ==operating mode==
parser.add_argument("--device_id", type=int, nargs='+', default=0, required=True, help="cuda device id")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--print_freq", type=float, default=100., help="Unit: iteration, freq of printing training status")
parser.add_argument("--ckpt_freq", type=int, default=10, help="Unit: epoch, freq of saving checkpoints")
parser.add_argument("--validate_freq", type=int, default=10, help="Unit: epoch, freq of doing validation")
parser.add_argument("--gan_pre_trained", action="store_true", help="called if you want to keep training GAN's ckpt")
parser.add_argument("--bm_version", required=True, choices=["ori", "re"], help="which version of aflw2000 gt you want to varify")
parser.add_argument("--seed", type=int)

args = parser.parse_args()

# logging configration
logging.basicConfig(
    # format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
    format='[%(levelname)s]: %(message)s',
    level=logging.INFO,
    # handlers=[
    #     logging.FileHandler(os.path.join(args.log_dir, "logging.log"), mode='w'),
    #     logging.StreamHandler()
    # ]
)

# # Tensorboard writer
# layout = {
#     "Generator": {
#         "loss": ["Multiline", ["G_loss_L1", "G_loss_D", "G_loss"]],
#     },
# }

# writer.add_custom_scalars(layout)

def train(G,D,OptimG, OptimD, SchedulerG, SchedulerD, ckpt, dataloader, criterion, face_wmask, v_set):
    # ==training loop==
    # - Discriminator loss : ℒ_𝐺𝑒𝑛(𝐺,𝐷)= 𝐸(𝑥∼𝑃_𝑑𝑎𝑡𝑎(𝑥))[−‖(𝑥,y)−𝐷(𝑥,y)‖1]+𝐸(𝑥∼𝑃_𝑑𝑎𝑡𝑎(𝑥))[−‖(𝑥,𝐺(𝑥))−𝐷(𝑥,𝐺(𝑥))‖1]
    # - Generator loss : ℒ_𝐺𝑒𝑛(𝐺,𝐷)= 𝐸(𝑥∼𝑃_𝑑𝑎𝑡𝑎(𝑥))[−‖(𝑥,𝐺(𝑥))−𝐷(𝑥,𝐺(𝑥))‖1]
    
    # ==Device configuration==
    device_id = ",".join(str(num) for num in args.device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ==TensorBoard==
    writer = SummaryWriter(args.tblog_dir)
    
    # for ploting model structure
    with torch.no_grad():
        G.eval()
        G_in = torch.rand((3, 3, 256, 256)).to(device)
        G_out = G(G_in)
        writer.add_graph(G.module, G_in)
        
        D.eval()
        D_in = torch.cat((G_in,G_out), dim=1)
        D_out = D(D_in)
        writer.add_graph(D.module, D_in)
    
    num_steps = len(dataloader) # Interations in one Epoch
    epoch_runned = ckpt["epoch"] if args.gan_pre_trained else 0
    global_step = ckpt["global_step"] if args.gan_pre_trained else 0 #unit: iteration, for tensorboard recording
    
    
    # margin = ckpt['margin'] if args.gan_pre_trained else 0.0036 # experience number from training dataset performance
    k_t = 0.
    
    gamma_real = args.gamma if args.gamma <= 1 else 1.
    gamma_fake = 1.         if args.gamma <= 1 else 1./args.gamma
    
    for epoch in range(epoch_runned, args.epochs):
        epoch = epoch + 1 # My prefer style is start from 1.
        G.train() # Switch to training mode
        D.train() # Switch to training mode
        
        start_time = time.process_time() # Recording time/epoch for time estimation.
        
        running_G_loss_L1 = 0.0 # just set for recording
        running_G_loss_D = 0.0 # just set for recording
        running_D_real_loss = 0.0 # just set for recording
        running_D_loss = 0.0 # just set for recording
        running_k_t = 0.0 # just set for recording
        
        for step, batched_data in enumerate(dataloader):
            
            step = step + 1
            global_step = global_step + 1
            
            image, uvmap = batched_data[0], batched_data[1]
            
            # ===================== training G =====================
            OptimG.module.zero_grad()
            
            G_in = image.to(device)
            G_GT = uvmap.to(device) # generator's ground truth
            
            G_out = G(G_in)
            
            D_in = torch.cat((G_in,G_out), dim=1)
            D_out = D(D_in)
            
            
            G_loss_L1 = criterion(G_out*face_wmask, G_GT*face_wmask)
            G_loss_D = criterion(D_out, D_in)
            G_loss = G_loss_L1 * args.l1_weight + G_loss_D * args.gan_weight
            
            # Backward and optimize
            G_loss.backward()
            OptimG.module.step()
            
            # Just for record
            running_G_loss_L1 += G_loss_L1.item()
            running_G_loss_D += G_loss_D.item()
            
            # ===================== training D =====================
            # margin way form EBGAN
            # if G_loss_D.item() > margin:
            #     OptimD.module.zero_grad()
            # else:
            #     D.module.neg_grad()
            
            # gamma-k_t way from BEGAN
            D.module.neg_grad(k_t)
            
            D_in = torch.cat((image, uvmap), dim=1).to(device)
            D_out = D(D_in)
            D_real_loss = criterion(D_out, D_in)
            
            # Backward and optimize
            D_real_loss.backward()
            OptimD.module.step()
            
            
            # D_fake_loss = torch.zeros(1) if G_loss_D > margin else margin - G_loss_D
            # D_loss = D_real_loss.item() + D_fake_loss.item() 
            D_fake_loss = G_loss_D 
            D_loss = D_real_loss.item() + k_t * D_fake_loss.item() 
            
            # Just for record
            running_D_real_loss += D_real_loss.item()
            running_D_loss += D_loss
            
            
            
            # k_t Record and Update
            running_k_t += k_t
            
            k_t += args.lambda_k * (gamma_real * D_real_loss.item() - gamma_fake * D_fake_loss.item()) 
            k_t = max(0., (min(1., k_t)))
            
            # Setting by args, printing frequency.
            if step % args.print_freq == 0:
                print(f'Epoch [{epoch}/{args.epochs}], ' \
                    + f'Step [{step:04d}/{num_steps}], ' \
                    \
                    + f'G_loss_L1: {(running_G_loss_L1 / args.print_freq):.4f}, ' \
                    + f'G_loss_D: {(running_G_loss_D / args.print_freq):.4f}, ' \
                    + f'G_loss: {((running_G_loss_L1 + running_G_loss_D)/args.print_freq):.4f}, '
                    \
                    + f'D_fake_loss: {(running_G_loss_D / args.print_freq):.4f}, ' \
                    + f'D_real_loss: {(running_D_real_loss / args.print_freq):.4f}, ' \
                    + f'D_loss: {(running_D_loss/args.print_freq):.4f}, ' \
                    \
                    + f'k_t: {running_k_t/args.print_freq:.4f} ' \
                    + f'balance: {(gamma_real * running_D_real_loss - gamma_fake * running_G_loss_D) / args.print_freq:.4f}')

                
                writer.add_scalars("Generator_loss",
                                {
                                    "G_loss_L1": (running_G_loss_L1 / args.print_freq),
                                    "G_loss_D": (running_G_loss_D / args.print_freq),
                                    "G_loss": ((running_G_loss_L1 + running_G_loss_D)/args.print_freq)
                                },
                                global_step)

                writer.add_scalars("Discriminator_loss",
                                {
                                    "D_fake_loss": (running_G_loss_D / args.print_freq),
                                    "D_real_loss": (running_D_real_loss / args.print_freq),
                                    "D_loss": (running_D_loss/args.print_freq),
                                    # "Margin": margin,
                                },
                                global_step)
                
                writer.add_scalar("k_t", running_k_t/args.print_freq, global_step=global_step)
                writer.add_scalar("balance", (args.gamma * running_D_real_loss - running_G_loss_D) / args.print_freq , global_step=global_step)
                
                
                running_G_loss_D = 0.0
                running_G_loss_L1 = 0.0
                running_D_loss = 0.0
                running_D_real_loss = 0.0
                running_k_t = 0.0

        SchedulerD.step()
        SchedulerG.step()
        # margin = margin * 0.99 if epoch % 10 == 0 else margin # 0.98 -> experience number
        
        # Estimate the remaining time of traning process
        op_time = time.process_time() - start_time # Unit: second
        remain_sec = (args.epochs-epoch) * op_time
        hour = remain_sec // 3600
        minu = (remain_sec - hour*3600) // 60
        
        print(f"\nFor next epoch:")
        print(f"    G's Learning rate: {SchedulerG.get_last_lr()[0]:.6f}")
        print(f"    D's Learning rate: {SchedulerD.get_last_lr()[0]:.6f}")
        # print(f"    Margin: {margin:.6f}")
        print(f'\nTime spent (this epoch): {op_time//60}minute {op_time%60:.2f}sec')
        print(f'Time_remain: {hour}hour {minu}minute')
        

        # validate with benchmark dataset
        if epoch % args.validate_freq == 0:
            with torch.no_grad():
                G.eval()
                D.eval()
                
                G_in = v_set[:,:3,:,:].to(device)
                G_out = G(G_in).to(device)
                G_GT = v_set[:,3:,:,:].to(device)
                
                D_in = torch.cat((G_in,G_out), dim=1)
                D_out = D(D_in)
                
                G_loss_L1_v = criterion(G_out*face_wmask, G_GT*face_wmask)
                G_loss_D_v = criterion(D_out, D_in)
                
                # Record the validation loss.
                writer.add_scalars("Validation_loss",
                                {
                                    "G_loss_L1_v": G_loss_L1_v.item(),
                                    "G_loss_D_v": G_loss_D_v.item(),
                                    "G_loss_v": (G_loss_L1_v + G_loss_D_v).item()
                                },
                                global_step)
                
                # Record the validation image.
                validate = [val for pair in zip(G_in[:8], G_GT[:8], G_out[:8]) for val in pair]
                validate = torch.stack(validate)
                validate_grid = torchvision.utils.make_grid(validate, nrow=3*2, normalize=True)
                writer.add_image("Validate", validate_grid, epoch)
                torchvision.utils.save_image(validate_grid, f"{args.img_dir}Validate_{epoch}.jpg")
            
            logging.info(f"Validation Epoch:{epoch}, G_loss_L1_v: {G_loss_L1_v.item():.4f}, G_loss_D_v: {G_loss_D_v.item():.4f}")    
            logging.info(f"Generated Image by Discriminator in Epoch:{epoch} is stored!!")
        
        # model saving
        if epoch % args.ckpt_freq == 0:
            ckpt = {
                "epoch": epoch,
                "k_t": k_t,
                "global_step": global_step,
                "Generator": G.module.state_dict(),
                "optimG": OptimG.module.state_dict(),
                "schedulerG": SchedulerG.state_dict(),
                "Discriminator": D.module.state_dict(),
                "optimD": OptimD.module.state_dict(),
                "schedulerD": SchedulerD.state_dict()
            }
            torch.save(ckpt, args.ckpt_dir+"ckpt_epoch_"+str(epoch).zfill(3)+".pth")
            logging.info(f'Model Saved, Epoch: {epoch}')
        
        print("="*30)


def main():
    # ==File io==
    # ckpt
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
        
    # generated_image
    if not os.path.exists(args.img_dir):
        os.makedirs(args.img_dir)
    
    # tensorboard
    if not os.path.exists(args.tblog_dir):
        os.makedirs(args.tblog_dir)
        
    # == for recording your setting this time, avoid some accident happend...==
    parent_dir = os.path.join(args.img_dir, os.pardir)
    
    if args.mode == "train":
        with open(os.path.join(parent_dir, "options.json"), "w") as f:
            f.write(json.dumps(vars(args), sort_keys=False, indent=4))
    
    # == custom face mask ==
    face_wmask = Image.open("face_weight_mask.jpg")
    face_wmask = transforms.ToTensor()(face_wmask)
    # face_weights_mask = face_weights_mask * 16.

    # ==Dataset and Dataloader==
    # magic number of Normalization mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    aug_set = [
        RandomErase_wk(seed=args.seed, max_num=4),
        # transforms.RandomErasing(p=1, scale=(0.02,0.3), ratio=(0.3,3.3), value='random'),
        # ChannelScale_wk(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(5, 2),
    ]
    
    # augmentation probability
    aug_prob = 0.5 if args.mode == 'train' else 0.0
    
    # for Image
    transform = transforms.Compose([
        transforms.ToTensor(), # Normalize tensors(C,H,W, uint8) to [0, 1](float32)
        transforms.Resize(size=[256,256]),
        transforms.RandomApply(aug_set, p=aug_prob),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    
    # for UVMap_npy
    target_transform = transforms.Compose([
        npy2tensor_wk(), # channel switch, divided by 255., change from ndarray to tensor
        transforms.Resize(size=[256,256]),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]), # [0, 1] -> [-1, 1] (approximately)
    ])
    
    # tensor [-1,1 CHW] -> Image[0,1 HWC]
    inverse_transform = transforms.Compose([
        transforms.Normalize([0,0,0],[2,2,2]),
        transforms.Normalize([-0.5,-0.5,-0.5],[1,1,1]),
        tensor2npy_wk()
    ])
    
    dataset = I2P_train(args.input_dir, transform=transform, target_transform=target_transform)
    benchmark = I2P_bm(args.bchm_dir, transform=transform, target_transform=target_transform)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size*len(args.device_id), shuffle=True, drop_last=True, num_workers=4)
    
    np.random.seed(args.seed)
    validate_idx = np.random.choice(len(benchmark), size=args.batch_size*len(args.device_id), replace=False)
    validate_set = [benchmark[idx] for idx in validate_idx] # list(size,) of tuple(2,) of tensors(3,256,256)
    v_image, v_uvmap = zip(*validate_set) # 2 list(size,) of tensor(3,256,256)
    v_image = torch.stack(v_image) # (size, 3, H, W)
    v_uvmap = torch.stack(v_uvmap) # (size, 3, H, W)
    v_set = torch.cat((v_image, v_uvmap), dim=1) # (size, 6, H, W)
    
    criterion = nn.L1Loss()
    
    logging.info("Operation Mode: "+ args.mode)
    
    if args.mode == 'train':
        # ==Device configuration==
        device_id = ",".join(str(num) for num in args.device_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Device: " + str(device))
        
        # == face weighted mask
        face_wmask = face_wmask.to(device)
        
        # ==Model (Unet, Generator, Discriminator)==
        G = Generator(in_channels=3 ,args= args).to(device)
        D = Discriminator(in_channels=3*2 ,args= args).to(device)
        
        # ==Optimizer and Scheduler==
        OptimD = Adam(D.parameters(), lr=args.lr, betas=(args.beta1,args.beta2))
        OptimG = Adam(G.parameters(), lr=args.lr, betas=(args.beta1,args.beta2))
        
        # learning rate scheduler
        # SchedulerD = lr_scheduler.StepLR(OptimD, step_size=30, gamma=0.8) # Unit of step size: Epoch
        # SchedulerG = lr_scheduler.StepLR(OptimG, step_size=30, gamma=0.8)
        
        SchedulerD = lr_scheduler.ExponentialLR(OptimD, gamma=0.9924)
        SchedulerG = lr_scheduler.ExponentialLR(OptimG, gamma=0.9924)
        
        # load the GAN ckpt if exists else load D ckpt
        latest_ckpt = "../Log/D_training_log/img_npy_3/ckpt/ckpt_epoch_300.pth"
        
        ckpt = load_newest_ckpt(args.ckpt_dir, device) if args.gan_pre_trained \
            else torch.load(latest_ckpt, map_location=device)
        
        # I use the same key for storing D related parameters, so it's no need to use "if" to judge.
        D.load_state_dict(ckpt["Discriminator"])    
        OptimD.load_state_dict(ckpt['optimD'])
        SchedulerD.load_state_dict(ckpt['schedulerD'])
        
        if args.gan_pre_trained: # if you are training GAN in halfway and accidentally be interrupted, ckpt may be helpful.
            G.load_state_dict(ckpt["Generator"])
            OptimG.load_state_dict(ckpt['optimG'])
            SchedulerG.load_state_dict(ckpt['schedulerG'])
            logging.info(f"trained Generator and Discriminator is loaded")
        else:
            logging.info(f"Pre-trained Discriminator from {latest_ckpt} is loaded")    
        
        # wrap in "Dataparallel" for using multi-GPU training
        D = nn.DataParallel(D, device_ids=args.device_id).to(device)
        G = nn.DataParallel(G, device_ids=args.device_id).to(device)
        OptimD = nn.DataParallel(OptimD, device_ids=args.device_id).to(device)
        OptimG = nn.DataParallel(OptimG, device_ids=args.device_id).to(device)

        logging.info(f"GPU(s) specified: {device_id}")
        train(G, D, OptimG, OptimD, SchedulerG, SchedulerD, ckpt, dataloader, criterion, face_wmask, v_set)
    
    elif args.mode == "export":
        
        # benchmark modify
        if args.bm_version == "re":
            in_dir = "/home/vlsilab/Dataset/aflw2000_data/wk_256/"
            benchmark = I2P_bm_re(in_dir, transform)
        
        # == load trained model G ==
        device = torch.device("cpu")
        logging.info("Device: cpu")
        
        # ==load uvmap_kpt location==
        #! Original data, is (y, x) instead of (x, y)
        uv_kpt_ind = np.loadtxt('../Image/uv_kpt_ind.txt').astype(np.int32) # 2 x 68 get kpt
        
        # ==loop for ckpt benchnmarking==
        for i in [29]:
            
            i = int((i+1)*10)
            ckpt = torch.load(f"../Log/GAN_training_log/img_npy_4/ckpt/ckpt_epoch_{i:03d}.pth", map_location=device)
            
            G = Generator(in_channels=3 ,args= args).to(device)
            G.load_state_dict(ckpt["Generator"])
            G.to(device)
            G.eval()
            
            subprocess.run(["rm", "../Log/GAN_training_log/img_npy_4/Name.log"])
            subprocess.run(["rm", "../Log/GAN_training_log/img_npy_4/NME_2D_68.log"])
            subprocess.run(["rm", "../Log/GAN_training_log/img_npy_4/NME_3D_68.log"])
            
            Name_list, NME_2D_68_list, NME_3D_68_list = get_align68_losses(G, benchmark, inverse_transform, uv_kpt_ind, parent_dir, bm_version=args.bm_version)
        
            # print(f"NME_3D_68 Error: Mean:{np.mean(NME_3D_68_list)*100:.3f}%, Var:{np.var(NME_3D_68_list):.5f}")
            print(f"NME_2D_68 Error: Mean:{np.mean(NME_2D_68_list)*100:.3f}%, Var:{np.var(NME_2D_68_list):.5f}")
    
            continue
        
        #     # sorted by NME 2d error from high to low
        #     assemble = list(zip(Name_list, NME_2D_68_list)) # pitch yaw roll
        #     assemble = sorted(assemble, key= lambda x: np.abs(x[1]), reverse=True)
        #     Name_list, NME_2D_68_list = zip(*assemble)
            
        #     # w means the worst!!
        #     w_name = Name_list[:48]
        #     w_set = [benchmark[benchmark.get_idx_by_name(name)] for name in w_name] # list(size,) of tuple(2,) of tensors(3,256,256)
        #     w_image, w_uvmap = zip(*w_set) # 2 list(size,) of tensor(3,256,256)
        #     w_image = torch.stack(w_image) # (size, 3, H, W)
        #     w_uvmap = torch.stack(w_uvmap) if args.bm_version == "ori" else np.transpose(np.stack(w_uvmap, axis=0), (0,2,1)) # (size, 3, H, W)  or (size, 68, 2)
        #     # w_set = torch.cat((w_image, w_uvmap), dim=1) # (size, 6, H, W)
            
        #     # == data process ==
        #     with torch.no_grad():
        #         G_in = w_image.to(device)
        #         G_out = G(G_in)
            
        #     G_in = inverse_transform(G_in)
        #     G_out = inverse_transform(G_out) # Go: Generator output
        #     G_GT = inverse_transform(w_uvmap) if args.bm_version == "ori" else w_uvmap # GT: ground truth
            
        #     # == Export the grid of alignment from Generator and Ground Truth==
        #     grid_of_alignment(G_in, G_out, uv_kpt_ind, f"../Image/img_npy_4/worst_GAN_epoch{i}.jpg",  error_list = NME_2D_68_list[:48])
        #     grid_of_alignment(G_in, G_GT, uv_kpt_ind, f"../Image/img_npy_4/worst_GT_epoch{i}.jpg", bm_version=args.bm_version)
        
        
        sys.exit()
        
        
        # ckpt = load_newest_ckpt(args.ckpt_dir, device)
        ckpt = torch.load("../Log/GAN_training_log/img_npy_4/ckpt/ckpt_epoch_230.pth", map_location=device)
        
        G = Generator(in_channels=3 ,args= args).to(device)
        G.load_state_dict(ckpt["Generator"])
        G.to(device)
        G.eval()
        
        D = Discriminator(in_channels=3*2 ,args= args).to(device)
        D.load_state_dict(ckpt["Discriminator"])
        D.to(device)
        D.eval()
        
        
        
        # == data process ==
        with torch.no_grad():
            G_in = v_image.to(device)
            G_out = G(G_in)
        
        G_in = inverse_transform(G_in)
        G_GT = inverse_transform(v_uvmap) # GT: ground truth
        G_out = inverse_transform(G_out) # Go: Generator output
        
        # check the value of range between GT and Gout
        print("the value range of G_GT is :", np.max(G_GT), np.min(G_GT)) 
        print("the value range of G_out is :", np.max(G_out), np.min(G_out))
        
        # == Export the grid of alignment from Generator and Ground Truth==
        os.makedirs("../Image/img_npy_4", exist_ok=True)
        grid_of_alignment(G_in, G_out, uv_kpt_ind, "../Image/img_npy_4/GAN_bm_ori_random.jpg")
        grid_of_alignment(G_in, G_GT, uv_kpt_ind, "../Image/img_npy_4/GT_bm_ori_random.jpg")
        
        # == Calculate the NME and statistic of CDE ==
        if args.bm_version == 'ori':
            logging.info(f"Calculating the reconstruction error of img+GTpos")
            test_img_with_GTpos_losses = get_dataset_losses(D, criterion, benchmark, os.path.join(parent_dir,"test_img_with_GTpos.log"), device)
            
            logging.info(f"Calculating the reconstruction error of img+Goutpos")
            test_img_with_Goutpos_losses = get_dataset_losses(D, criterion, benchmark, os.path.join(parent_dir,"test_img_with_Goutpos.log"), device, G=G)

            print(f"Reconstruction Error of GTpos: Mean:{np.mean(test_img_with_GTpos_losses):.6f}, Variance:{np.var(test_img_with_GTpos_losses):.8f}")
            print(f"Reconstruction Error of Gopos: Mean:{np.mean(test_img_with_Goutpos_losses):.6f}, Variance:{np.var(test_img_with_Goutpos_losses):.8f}")
        
        # == calculate the NME of each validation data in benchmark dataset
        #! already sorted by name!!!
        Name_list, NME_2D_68_list, NME_3D_68_list = get_align68_losses(G, benchmark, inverse_transform, uv_kpt_ind, "../Log/GAN_training_log/img_npy_4/", bm_version=args.bm_version)
        
        # == Analysis of alignment error
        with open("./AFLW2000-3D_fname_pose_gimbal_lock", 'rb')as fp:
            fname_angle_list = pickle.load(fp)
        #! already sorted by name!!!
        fname_list, angle_list = zip(*fname_angle_list) # 2 list(size,) of tensor(3,256,256)
        
        assert(Name_list == fname_list)
        
        
        
        
    
        # sorted by NME 2d error from high to low (you can alter "key" below for the order you want)
        assemble = list(zip(Name_list, NME_2D_68_list, angle_list)) # pitch yaw roll
        assemble = sorted(assemble, key= lambda x: np.abs(x[1]), reverse=True)
        Name_list, NME_2D_68_list, angle_list = zip(*assemble)
        
        Name_list = list(Name_list)
        NME_2D_68_list = list(NME_2D_68_list)
        angle_list = list(angle_list)
        
        # # avoid some bad benchmark
        # avoid_list = ["image03871", "image03101"]
        # avoid_list = [Name_list.index(name) for name in avoid_list]
        
        # for i in sorted(avoid_list, reverse=True):
        #     Name_list.pop(i)
        #     NME_2D_68_list.pop(i)
        #     angle_list.pop(i)
        
        print(f"NME_3D_68 Error: Mean:{np.mean(NME_3D_68_list)*100:.3f}%, Var:{np.var(NME_3D_68_list):.3f}")
        print(f"NME_2D_68 Error: Mean:{np.mean(NME_2D_68_list)*100:.3f}%, Var:{np.var(NME_2D_68_list):.3f}")
    

        # == Analysis ==
        ## gimbol lock + roll 0~60~120~180
        # AFLW2000_3D_pitch = np.array([1814, 128,  58,])
        # AFLW2000_3D_yaw = np.array([1313, 383, 304,])
        # AFLW2000_3D_roll = np.array([1931, 60, 9,])
        
        ## == statistic by zy ==
        # AFLW2000_3D_pitch = np.array([1813, 126, 61])
        # AFLW2000_3D_yaw = np.array([1312, 383, 305])
        # AFLW2000_3D_roll = np.array([1745, 184, 71])
        
        ## gimbol lock
        # AFLW2000_3D_pitch = np.array([1814, 128,  58,])
        # AFLW2000_3D_yaw = np.array([1313, 383, 304,])
        # AFLW2000_3D_roll = np.array([1746, 185,  69,])
        
        dict = {
            "pitch": (0, [30, 60, 90] , [1814, 128,  58,]),
            "yaw"  : (1, [30, 60, 90] , [1313, 383, 304] ),
            'roll' : (2, [60, 120, 180], [1931, 60, 9,]   )
        }
        
        choose = dict.keys()
        
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
            print("="*20)
    
        
            fig = plt.figure(figsize=(50,20))
        
        
            plt.bar(list(range(2000)),NME_2D_68_list, color=color_list)
            plt.title(c, fontsize=40)
        
            colors = {f'0~{dict[c][1][0]}':'green', f'{dict[c][1][0]}~{dict[c][1][1]}':'blue', f'{dict[c][1][1]}~{dict[c][1][2]}':'red'}         
            labels = list(colors.keys())
            handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
            plt.legend(handles, labels, fontsize=40)
        
            plt.savefig(f"../Image/img_npy_4/NME_2D_68_{c}.png")

        
        # # w means the worst!!
        # w_name = Name_list[:48]
        # w_set = [benchmark[benchmark.get_idx_by_name(name)] for name in w_name] # list(size,) of tuple(2,) of tensors(3,256,256)
        # w_image, w_uvmap = zip(*w_set) # 2 list(size,) of tensor(3,256,256)
        # w_image = torch.stack(w_image) # (size, 3, H, W)
        # w_uvmap = torch.stack(w_uvmap) # (size, 3, H, W)
        # w_set = torch.cat((w_image, w_uvmap), dim=1) # (size, 6, H, W)
        
        # # == data process ==
        # with torch.no_grad():
        #     G_in = w_image.to(device)
        #     G_out = G(G_in)
        
        # G_in = inverse_transform(G_in)
        # G_GT = inverse_transform(w_uvmap) # GT: ground truth
        # G_out = inverse_transform(G_out) # Go: Generator output
        
        # # == Export the grid of alignment from Generator and Ground Truth==
        # grid_of_alignment(G_in, G_out, uv_kpt_ind, "../Image/img_npy_4/worst_grid_by_GAN.jpg", angle_list = angle_list[:48],  error_list = NME_2D_68_list[:48], name_list=Name_list[:48])
        # grid_of_alignment(G_in, G_GT, uv_kpt_ind, "../Image/img_npy_4/worst_grid_of_GT.jpg", angle_list = angle_list[:48], name_list=Name_list[:48], bm_version=args.bm_version)
        
        
        

        
        
        # How about fix the roll angle
        w_name = Name_list[:48]
        w_set = [benchmark[benchmark.get_idx_by_name(name)] for name in w_name] # list(size,) of tuple(2,) of tensors(3,256,256)
        
        _, w_uvmap = zip(*w_set) # 2 list(size,) of tensor(3,256,256)
        
        w_uvmap = torch.stack(w_uvmap)
        w_uvmap = inverse_transform(w_uvmap)
        
        def rotmatrix(radian):
            R = np.array([
                [np.cos(radian), np.sin(-radian), 0],
                [np.sin(radian),  np.cos(radian), 0],
                [             0,               0, 1]
            ])
            return R
        
        w_uvmap = [np.matmul((w_uvm - np.array([128,128,0])),rotmatrix(np.deg2rad(angle[2]))) + np.array([128,128,0]) for w_uvm, angle in zip(w_uvmap, angle_list[:48])]
        
        w_image = [Image.open("/home/vlsilab/Dataset/Img2Pos_test/AFLW2000_all-crop/" + os.path.join(name,name) + "_cropped.jpg") for name in w_name]
        w_image = [transform(w_img.rotate(angle[2])) for w_img , angle in zip(w_image, angle_list[:48])]
        w_image = torch.stack(w_image)
        
        G_in = w_image
        G_out = G(G_in)
        G_in = inverse_transform(G_in)
        G_out = inverse_transform(G_out)
        
        w_image = inverse_transform(w_image)
        
        NME_2D_68_fix_list = []
        for name, g_out, uvmap in zip(Name_list[:48], G_out, w_uvmap):
            G_out_68 = get_landmarks(g_out, uv_kpt_ind) # shape: (68,3)
            G_GT_68  = get_landmarks(uvmap, uv_kpt_ind) # shape: (68,3)
            L2_loss_2D_68 = np.mean(np.sqrt(np.sum(np.square(G_out_68[:,:2] - G_GT_68[:,:2]), axis=1)))

            _, bbox, _ = benchmark.__get_name_bbox_kpt__(benchmark.get_idx_by_name(name))
            
            normalization_factor = np.sqrt(np.prod(bbox[1] - bbox[0])) # sqrt of bbox area
            NME_2D_68 = L2_loss_2D_68/normalization_factor    
            NME_2D_68_fix_list.append(NME_2D_68)
            
        grid_of_alignment(G_in, G_out, uv_kpt_ind, "../Image/img_npy_4/fix_roll_GAN.jpg", error_list = NME_2D_68_fix_list)
        grid_of_alignment(w_image, w_uvmap, uv_kpt_ind, "../Image/img_npy_4/fix_roll_GT.jpg")
        
        print(f"improvment percentage:{(sum(NME_2D_68_list[:48]) - sum(NME_2D_68_fix_list))/len(NME_2D_68_fix_list)*100:.2f}%")

if __name__ == '__main__':
    main()
