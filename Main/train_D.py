import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
import logging
import time
import sys

from torch.utils.tensorboard import SummaryWriter

from Model.Discriminator import Discriminator
from Model.Generator import Generator
from Data_process import *
from Utils import *

# parser configuration
parser = argparse.ArgumentParser()

# file io
parser.add_argument("--input_dir", required=True, help="path to folder containing training data")
parser.add_argument("--bchm_dir", required=True, help="path to folder containing benchmark data")
parser.add_argument("--ckpt_dir", required=True, help="path to folder containing training data")
parser.add_argument("--img_dir", required=True, help="path to folder containing training data")
parser.add_argument("--tblog_dir", required=True, help="path to folder containing training data")

# Model & training configuraiton
parser.add_argument("-ngf", "--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("-ndf", "--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="number of images in batch")
parser.add_argument("--beta1", type=float, default=0.9 , help="Adam's Hyper parameter")
parser.add_argument("--beta2", type=float, default=0.99, help="Adam's Hyper parameter")
parser.add_argument("--lr", type=float, default=0.00005, help="initial learning rate for adam")
parser.add_argument("--decay_step", type=int, default=30, help="Step(unit:epoch) to decay learning rate")

# Operation mode
parser.add_argument("--device_id", type=int, nargs='+', default=0, required=True, help="cuda device id")
parser.add_argument("--mode", default='train', choices=["train", "test", "export"])
parser.add_argument("--print_freq", type=float, default=100., help="Unit: iteration, freq of printing training status")
parser.add_argument("--ckpt_freq", type=int, default=10, help="Unit: epoch, freq of saving checkpoints")
parser.add_argument("--validate_freq", type=int, default=10, help="Unit: epoch, freq of doing validation")
parser.add_argument("--seed", type=int, default=0, help="for fixing the random logic")
parser.add_argument("--pre_trained", action="store_true", help="called if you want to keep training \"trained model\" to maximun epoch")
args = parser.parse_args()

logging.basicConfig(
    format='[%(levelname)s]: %(message)s',
    level=logging.INFO,
)



def train(args, D, criterion, OptimD, SchedulerD, dataloader, v_set):
    
    # ==Device configuration==
    device_id = ",".join(str(num) for num in args.device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"GPU(s) specified: {device_id}")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ==TensorBoard==
    writer = SummaryWriter(args.tblog_dir)
    
    # ==Pretrained process==
    if args.pre_trained:
        ckpt = load_newest_ckpt(args.ckpt_dir, device)
        # reload the state from ckpt file
        D.load_state_dict(ckpt['Discriminator'])
        OptimD.load_state_dict(ckpt['optimD'])
        SchedulerD.load_state_dict(ckpt['schedulerD'])
    
    
    D = nn.DataParallel(D, device_ids=args.device_id)
    OptimD = nn.DataParallel(OptimD, device_ids=args.device_id)
    D.to(device)
    OptimD.to(device)
    
    epoch_runned = ckpt['epoch'] if args.pre_trained else 0 # if you want to train Model which is trained halfway through.
    
    # ==training loop==
    num_steps = len(dataloader) # Interations in one Epoch
    global_step = ckpt["global_step"] if args.pre_trained else 0 #unit: iteration, for tensorboard recording
    
    for epoch in range(epoch_runned, args.epochs):
        start_time = time.process_time() # Recording time/epoch for time estimation.
        
        epoch = epoch + 1 # My prefer style is start from 1.
        D.train() # Switch to training mode
        
        running_loss = 0.0 # just set for recording
        
        writer.add_scalar("Learning rate", SchedulerD.get_last_lr()[0], global_step)
        
        for step, batched_data in enumerate(dataloader):
            
            step = step + 1
            global_step = global_step + 1
            
            # Input process
            image, uvmap = batched_data[0] , batched_data[1]
            D_in = torch.cat((image, uvmap), dim=1).to(device) # N,C=6,H,W
            
            # forward
            D_out = D(D_in)
            D_loss = criterion(D_out, D_in)
            running_loss += D_loss.item()
            
            # Backward and optimize
            OptimD.module.zero_grad()
            D_loss.backward()
            OptimD.module.step()
            
            # printing frequency, to show training status.
            if step % args.print_freq == 0:
                print(f'Epoch [{epoch}/{args.epochs}],' \
                     + f'Step [{step:04d}/{num_steps}],' \
                     + f'D_loss: {running_loss/args.print_freq:.4f},')
                
                writer.add_scalar("D_loss", (running_loss/args.print_freq), global_step)
                running_loss = 0.0
        
        SchedulerD.step() # Step every epoch
        
        # Estimate the remaining time of traning process
        op_time = time.process_time() - start_time # Unit: second
        remain_sec = (args.epochs-epoch) * op_time
        hour = remain_sec // 3600
        min = (remain_sec - hour*3600) // 60
        
        print(f"\nLearning rate (for next epoch): {SchedulerD.get_last_lr()[0]:.6f}")
        print(f'Time spent (this epoch): {op_time//60}min {op_time%60:.2f}sec')
        print(f'Time_remain: {hour}hour {min}min')
        
        # validate region
        if epoch % args.validate_freq == 0:
            with torch.no_grad():
                D.eval()
                D_in = v_set.to(device)
                D_out = D(D_in)
                D_loss = criterion(D_out,D_in)
                
                writer.add_scalar("D_loss_v", (D_loss.item()), global_step=epoch)
                
                v_image = D_in[:4,:3,:,:]
                v_uvmap = D_in[:4,3:,:,:]
                v_image_ae = D_out[:4,:3,:,:]
                v_uvmap_ae = D_out[:4,3:,:,:]
              
                # Record the validation image (4 pairs, 4 images per pair).
                validate = [val for pair in zip(v_image, v_uvmap, v_image_ae, v_uvmap_ae) for val in pair]
                validate = torch.stack(validate)
                validate_grid = torchvision.utils.make_grid(validate, nrow=4, normalize=True)
                writer.add_image("Validate", validate_grid, epoch)
                torchvision.utils.save_image(validate_grid, f"{args.img_dir}Validate_{epoch}.jpg")
            
            logging.info(f"Validation Epoch:{epoch}, D_loss: {D_loss.item():.4f}")    
            logging.info(f"Generated Image by Discriminator in Epoch:{epoch} is stored!!")
        
        # model saving
        if epoch % args.ckpt_freq == 0:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "Discriminator": D.module.state_dict(), # DataParallel
                "optimD": OptimD.module.state_dict(), # DataParallel
                "schedulerD": SchedulerD.state_dict()
            }
            torch.save(ckpt, args.ckpt_dir+"ckpt_epoch_"+str(epoch).zfill(3)+".pth")
            logging.info(f'Model Saved, Epoch: {epoch}')
        
        print("="*30)    
            
    logging.info("== Training Completed ==")

def main():
    """
    I list what I have done in main function,
        - File IO process, if specified dir not exist, then create it.
        - Create some tranformation(100%) for loaded in Image(face pic, uvmap) and design some augmentation strategies(50%) for face pic.
        - Create object of Model, optimizer, scheduler, dataset, dataloader, etc.
        - According the needs (train, export, or test), run different function.
    """
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
    
    # ==Device configuration==
    device_id = ",".join(str(num) for num in args.device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ==Model (Unet, Discriminator)==
    D = Discriminator(in_channels=6, args=args)
    D.to(device)
    
    # ==Optimizer, Criterion, and Scheduler==
    criterion = nn.L1Loss()
    OptimD = Adam(D.parameters(), lr=args.lr, betas=(args.beta1,args.beta2))
    
    #* 2 choise of Scheduler
    # SchedulerD = lr_scheduler.StepLR(OptimD, step_size=args.decay_step, gamma=0.8) # Unit of step size: Epoch
    SchedulerD = lr_scheduler.ExponentialLR(OptimD, gamma=0.9924) # Unit of step size: Epoch
    
    
    # ==Dataset and Dataloader==
    # magic number of Normalization mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    aug_set = [
        RandomErase_wk(seed=args.seed),
        # transforms.RandomErasing(p=1, scale=(0.02,0.3), ratio=(0.3,3.3), value='random'),
        # ChannelScale_wk(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(5, 2),
    ]
    
    # augmentation probability
    aug_prob = 0.5 if args.mode == 'train' else 0.0
    
    # "normalize_mean": [0.485, 0.456, 0.406],
    # "normalize_std": [0.229, 0.224, 0.225],
    
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size*len(args.device_id), shuffle=True, drop_last=True, num_workers=4)
    
    benchmark = I2P_bm(args.bchm_dir, transform=transform, target_transform=target_transform)
    celeba = CelebA("/home/vlsilab/Dataset/celebA/img_align_celeba/", transform=transform)
    
    np.random.seed(args.seed)
    validate_idx = np.random.choice(len(benchmark), size=args.batch_size*len(args.device_id), replace=False)
    validate_set = [benchmark[idx] for idx in validate_idx] # list(size,) of tuple(2,) of tensors(3,256,256)
    v_image, v_uvmap = zip(*validate_set) # 2 list(size,) of tensor(3,256,256)
    v_image = torch.stack(v_image) # (size, 3, H, W)
    v_uvmap = torch.stack(v_uvmap) # (size, 3, H, W)
    v_set = torch.cat((v_image, v_uvmap), dim=1) # (size, 6, H, W)
    
    # # test region
    # (img_t, npy_t) = dataset[0]
    # print(type(img_t), type(npy_t))
    # print(img_t.shape, npy_t.shape)
    # print(torch.max(img_t), torch.min(img_t))
    # print(torch.max(npy_t), torch.min(npy_t))
    
    # img = inverse_transform(img_t)
    # npy = inverse_transform(npy_t)
    
    # print(type(img), type(npy))
    # print(np.max(img), np.min(img))
    # print(np.max(npy), np.min(npy))
    
    # plt.subplot(1,2,1)
    # plt.imshow(img)
    # plt.subplot(1,2,2)
    # plt.imshow(npy)
    # plt.show()
    
    logging.info("Operation Mode: "+ args.mode)
    
    if args.mode == "train":
        logging.info("Device: " + str(device))
        train(args, D, criterion, OptimD, SchedulerD, dataloader, v_set)
    
    #! need to be fixed    
    elif args.mode == "export":
        """
        export the CDE according to the dataset
        #! switch model you should noticed that...
        1. ckpt location
        2. key name of model stored in state_dict
        3. the storage location of the losses path
        4. which GPU are you specified: os.environ["CUDA_VISIBLE_DEVICES"] = ? (on the top of this code), you can only specify one GPU!! but the device always be "cuda:0"
        """
        # Use one GPU for inference some  information
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info("Device: " + str(device))

        tr_mean = []
        te_mean = []
        ot_mean = []
        
        tr_vari = []
        te_vari = []
        ot_vari = []
        
        
        for i in range(0,1):
            ckpt = load_newest_ckpt(args.ckpt_dir, device)
            # ckpt = torch.load(f"../Log/D_training_log/img_npy_2/ckpt/ckpt_epoch_{i*10:03d}.pth",map_location=device)
            D.load_state_dict(ckpt["Discriminator"])
            D.to(device).eval()
            
            G = Generator(in_channels=3 ,args= args).to(device)
            G.to(device).eval()
            
            
            # train_losses = get_dataset_losses(D, criterion, dataset, f"../Log/D_training_log/img_npy_2/loss/train_losses_D_{i*10:03d}.log", device)
            # test_losses = get_dataset_losses(D, criterion, benchmark, f"../Log/D_training_log/img_npy_2/loss/test_losses_D_{i*10:03d}.log", device)
            # other_losses = get_dataset_losses(D, criterion, celeba, f"../Log/D_training_log/img_npy_2/loss/celebA_losses_D_{i*10:03d}.log", device)
            
            # print(f"train_losses: len={min (10000,len(train_losses))} m={np.mean(train_losses):.5f}, v={np.var(train_losses):.8f}")
            # print(f"test_losses: len={min(2000,len(test_losses))} m={np.mean(test_losses):.5f}, v={np.var(test_losses):.8f}")
            # print(f"other_losses: len={min(10000,len(other_losses))} m={np.mean(other_losses):.5f}, v={np.var(other_losses):.8f}")
            
            # train_losses = get_dataset_losses(D, criterion, dataset, f"../Log/D_training_log/img_npy_2/loss/train_losses_D_all.log", device)
            test_losses = get_dataset_losses(D, criterion, benchmark, f"../Log/D_training_log/img_npy_2/loss/test_losses_D_all.log", device)
            # other_losses = get_dataset_losses(D, criterion, celeba, f"../Log/D_training_log/img_npy_2/loss/celebA_losses_D_all.log", device)
            
            # print(f"train_losses: len={len(train_losses)} m={np.mean(train_losses):.5f}, v={np.var(train_losses):.8f}")
            print(f"test_losses: len={len(test_losses)} m={np.mean(test_losses):.5f}, v={np.var(test_losses):.8f}")
            # print(f"other_losses: len={len(other_losses)} m={np.mean(other_losses):.5f}, v={np.var(other_losses):.8f}")
            
            generator_losses = get_dataset_losses(D, criterion, benchmark, f"../Log/D_training_log/img_npy_2/loss/test_withG_losses_D_all.log", device, G=G)
            print(f"generator_losses: len={len(generator_losses)} m={np.mean(generator_losses):.5f}, v={np.var(generator_losses):.8f}")
            
            # tr_mean.append(np.mean(train_losses))
            # te_mean.append(np.mean(test_losses))
            # ot_mean.append(np.mean(other_losses))
            
            # tr_vari.append(np.var(train_losses))
            # te_vari.append(np.var(test_losses))
            # ot_vari.append(np.var(test_losses))
            
        #     print(f"progress:{i}\r")

        # plt.subplot(2,1,1)
        # plt.plot(range(30), tr_mean, 'r', label="train_mean")
        # plt.plot(range(30), te_mean, 'g', label="test_mean")
        # plt.plot(range(30), ot_mean, 'b', label="celeba_mean")
        # plt.legend()
        
        # plt.subplot(2,1,2)
        # plt.plot(range(30), tr_vari, 'r', label="train_vari")
        # plt.plot(range(30), te_vari, 'g*', label="test_vari")
        # plt.plot(range(30), ot_vari, 'b', label="celeba_vari")
        
        # plt.legend()
        # plt.savefig("Energy_surface.png")
        
        # plot_CDE(("train","test",),(train_losses, test_losses))
        # plot_PDE(("train",), (train_losses,))
        
    # elif args.mode == "test":
    #     test()

if __name__ == "__main__":
    main()
