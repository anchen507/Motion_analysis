import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
import cv2
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from model import SDNet,TRNet, FRNet
from dataset import *
from utils.misc_utils import AverageMeter
from utils.evaluation import psnr
from M_PWCNet.get_model import get_model
from self_pretraining_loss.get_loss import get_loss
from utils.warp_utils import flow_warp,optic_flow_compt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="PyTorch Motion_analysis")
parser.add_argument("--save", default='./log', type=str, help="Save path")
parser.add_argument("--resume_SDN", default="", type=str, help="Resume path (default: none)")
parser.add_argument("--resume_TRN", default="", type=str, help="Resume path (default: none)")
parser.add_argument("--resume_FRN", default="", type=str, help="Resume path (default: none)")
parser.add_argument("--loading_EME", default="", type=str, help="pretraing MEN model path (default: none)")
parser.add_argument("--train_dataset_dir", default='./../data/', type=str, help="train_dataset")
parser.add_argument("--batchSize", type=int, default=10, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=4e-4, help="Learning Rate, Default=4e-4")
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument("--step", type=int, default=6, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=6")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--noise_ival", nargs=2, type=int, default=[0, 55], help="Noise training interval")
parser.add_argument("--val_noiseL", type=float, default=30, help='noise level used on validation set')
parser.add_argument("--patch_size", type=int, default=128, help="Patch size")
parser.add_argument("--input_frame", type=int, default=5, help="input frames, Default=5")
parser.add_argument("--output_frame", type=int, default=1, help="output frames, Default=1")
parser.add_argument("--lamda", type=int, default=0.25, help="Balance for coarse denoising and refining")
parser.add_argument("--is_pretrain", type=bool, default=False, help="pretraing the MEN")
parser.add_argument("--pretrain_with_SDN", type=bool, default=True, help="pretraining MEN with pretrained CDN") #Need to load a pretrained CDN

device_ids = [0]
global opt, model
opt = parser.parse_args()
opt.val_noiseL /= 255.
opt.noise_ival[0] /= 255.
opt.noise_ival[1] /= 255.

# Normalize noise between [0, 1]

def train(train_loader, epoch_num):
    # load the state_dict

    SDN = SDNet()

    SDN = torch.nn.DataParallel(SDN, device_ids=device_ids).cuda()
    
    TRN = TRNet()

    TRN = torch.nn.DataParallel(TRN, device_ids=device_ids).cuda()
    
    FRN = FRNet()

    FRN = torch.nn.DataParallel(FRN, device_ids=device_ids).cuda()
    
    EME = get_model().cuda()

    epoch_state = 0
    loss_list = []
    psnr_list = []
    loss_epoch = []
    loss_iv_epoch =[]
    psnr_epoch = []

    if opt.resume_SDN:
        ckpt = torch.load(opt.resume_SDN)
        SDN.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
    if opt.resume_TRN:
        ckpt = torch.load(opt.resume_TRN)
        TRN.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
    if opt.resume_FRN:
        ckpt = torch.load(opt.resume_FRN)
        FRN.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
    if opt.loading_EME:
        ckpt = torch.load(opt.loading_EME)
        EME.load_state_dict(ckpt['state_dict'])
    epoch_state=3

    optimizer = torch.optim.Adam([   {'params':SDN.parameters(),'lr':opt.lr},
                                     {'params':TRN.parameters(),'lr':opt.lr},
                                     {'params':FRN.parameters(),'lr':opt.lr},
                                     {'params':EME.parameters(),'lr':opt.lr*0.1}])
    criterion_MSE = torch.nn.MSELoss(reduction='sum').cuda()
    flow_loss = get_loss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)
    with torch.no_grad():
         valid(SDN,TRN,EME,FRN, 0)

    for idx_epoch in range(epoch_state, epoch_num):
        for idx_iter, clean_seq in enumerate(train_loader):
            EME.train()
            SDN.train()
            TRN.train()
            FRN.train()
            clean_seq=Variable(clean_seq).cuda()
            N,_,L,H,W = np.shape(clean_seq)
            # std dev of each sequence
            
            stdn = torch.empty((N, 1, 1, 1, 1)).cuda().uniform_(opt.noise_ival[0], to=opt.noise_ival[1])
            stdn = stdn.cuda().view(N,1,1,1,1)
            # draw noise samples from std dev tensor
            noise = torch.zeros_like(clean_seq).cuda()
            noise = torch.normal(mean=noise, std=stdn.expand_as(noise))
            #define noisy input
            
            noisy_seq=clean_seq + noise


            gt = clean_seq[:,:,int(opt.input_frame/2-opt.output_frame/2):int(opt.input_frame/2+opt.output_frame/2),:,:]
            spatial_gt=clean_seq
            N,C,L,H,W = np.shape(clean_seq)
            noisy_seq = Variable(noisy_seq).cuda()
            [noisy_seq1,noisy_seq2,noisy_seq3,noisy_seq4,noisy_seq5] = torch.split(noisy_seq,1,dim=2)
            
            noisy_seq1 = noisy_seq1.view(N,C,H,W)
            noisy_seq2 = noisy_seq2.view(N,C,H,W)
            noisy_seq3 = noisy_seq3.view(N,C,H,W)
            noisy_seq4 = noisy_seq4.view(N,C,H,W)
            noisy_seq5 = noisy_seq5.view(N,C,H,W)
            
            deno_seq1 = SDN(noisy_seq1)
            deno_seq2 = SDN(noisy_seq2)
            deno_seq3 = SDN(noisy_seq3)
            deno_seq4 = SDN(noisy_seq4)
            deno_seq5 = SDN(noisy_seq5)
            
            wraped_deframe1, _, flow1 = optic_flow_compt(deno_seq1,deno_seq3,EME,flow_loss)
            wraped_deframe2, _, flow2 = optic_flow_compt(deno_seq2,deno_seq3,EME,flow_loss)
            wraped_deframe3              = deno_seq3
            wraped_deframe4, _, flow4 = optic_flow_compt(deno_seq4,deno_seq3,EME,flow_loss)
            wraped_deframe5, _, flow5 = optic_flow_compt(deno_seq5,deno_seq3,EME,flow_loss)

            wraped_deframe1=FRN(wraped_deframe1,deno_seq3,flow1)
            wraped_deframe2=FRN(wraped_deframe2,deno_seq3,flow2)
            wraped_deframe4=FRN(wraped_deframe4,deno_seq3,flow4)
            wraped_deframe5=FRN(wraped_deframe5,deno_seq3,flow5)

            wraped_deframe1 = wraped_deframe1.view(N,C,1,H,W)
            wraped_deframe2 = wraped_deframe2.view(N,C,1,H,W)
            wraped_deframe3 = wraped_deframe3.view(N,C,1,H,W)
            wraped_deframe4 = wraped_deframe4.view(N,C,1,H,W)
            wraped_deframe5 = wraped_deframe5.view(N,C,1,H,W)

            
            spatial_denoframes=torch.cat([deno_seq1.view(N,C,1,H,W ),deno_seq2.view(N,C,1,H,W ),deno_seq3.view(N,C,1,H,W ),deno_seq4.view(N,C,1,H,W ),deno_seq5.view(N,C,1,H,W )],dim=2)

            wraped_seqences=torch.cat([wraped_deframe1,wraped_deframe2,wraped_deframe3,wraped_deframe4,wraped_deframe5],dim=2)
            
            deno_seq=TRN(wraped_seqences)
            
            loss_spa = criterion_MSE(spatial_denoframes, spatial_gt)*0.2
            
            loss_ref = criterion_MSE(deno_seq, gt)

            loss = loss_ref+opt.lamda*loss_spa
            loss_epoch.append(loss.detach().cpu())
            loss_iv_epoch.append(loss_spa.detach().cpu())
            psnr_epoch.append(psnr(deno_seq, gt))
            optimizer.zero_grad()
            loss.backward()
            #print("grad before clip:"+str())
            optimizer.step()
            if idx_iter%10==0:
               loss_list.append(float(np.array(loss_epoch).mean()))
               psnr_list.append(float(np.array(psnr_epoch).mean()))
               print(time.ctime()[4:-5] + ' Epoch---%d, loss_hybid---%f, loss_spa---%f, PSNR---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(loss_iv_epoch).mean()), float(np.array(psnr_epoch).mean())))
               loss_epoch = []
               psnr_epoch = []
        scheduler.step()
        if (idx_epoch+1) % 5 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            psnr_list.append(float(np.array(psnr_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, loss_epoch---%f, loss_spa_epoch---%f, PSNR---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(loss_iv_epoch).mean()), float(np.array(psnr_epoch).mean())))
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': SDN.state_dict(),
            }, 
            save_path=opt.save, filename='SDN' +  '_epoch' + str(idx_epoch + 1) + '.pth.tar')

            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': FRN.state_dict(),
            }, 
            save_path=opt.save, filename='FRN' +  '_epoch' + str(idx_epoch + 1) + '.pth.tar')

            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': TRN.state_dict(),
            }, 
            save_path=opt.save, filename='TRN' +  '_epoch' + str(idx_epoch + 1) + '.pth.tar')
            
            '''save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': EME.state_dict(),
            },
            save_path=opt.save, filename='EME' +  '_epoch' + str(idx_epoch + 1) + '.pth.tar')''' 
            loss_epoch = []
            psnr_epoch = []
        if (idx_epoch+1) % 20 == 0:
            with torch.no_grad():
                 EME.eval()
                 SDN.eval()
                 TRN.eval()
                 FRN.eval()
                 valid(SDN,TRN,EME,FRN, idx_epoch)    



def valid(SDN,TRN,EME, FRN, idx_epoch):
    valid_set = ValidSetLoader(opt.train_dataset_dir, patch_size=opt.patch_size, input_frame=opt.input_frame)
    valid_loader = DataLoader(dataset=valid_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    psnr_list = []
    avg_time=0
    print('Evaluating........')
    for idx_iter, clean_seq in enumerate(valid_loader):
        clean_seq = Variable(clean_seq).cuda()
        gt=clean_seq[:,:,int(opt.input_frame/2-opt.output_frame/2):int(opt.input_frame/2+opt.output_frame/2),:,:]
        noise = torch.empty_like(clean_seq).normal_(mean=0, std=opt.val_noiseL).to(torch.device('cuda'))
        noisy_clean_seq = clean_seq + noise
        start_time=time.time()
        N,C,L,H,W=np.shape(gt)
        [noisy_seq1,noisy_seq2,noisy_seq3,noisy_seq4,noisy_seq5]=torch.split(noisy_clean_seq,1,dim=2)
        noisy_seq1=noisy_seq1.view(N,C,H,W)
        noisy_seq2=noisy_seq2.view(N,C,H,W)
        noisy_seq3=noisy_seq3.view(N,C,H,W)
        noisy_seq4=noisy_seq4.view(N,C,H,W)
        noisy_seq5=noisy_seq5.view(N,C,H,W)
        deno_seq1 = SDN(noisy_seq1)
        deno_seq2 = SDN(noisy_seq2)
        deno_seq3 = SDN(noisy_seq3)
        deno_seq4 = SDN(noisy_seq4)
        deno_seq5 = SDN(noisy_seq5)
        wraped_deframe1, _, flow1 = optic_flow_compt(deno_seq1,deno_seq3,EME,flow_loss=None)
        wraped_deframe2, _, flow2 = optic_flow_compt(deno_seq2,deno_seq3,EME,flow_loss=None)
        wraped_deframe3              = deno_seq3
        wraped_deframe4, _, flow4 = optic_flow_compt(deno_seq4,deno_seq3,EME,flow_loss=None)
        wraped_deframe5, _, flow5 = optic_flow_compt(deno_seq5,deno_seq3,EME,flow_loss=None)

        wraped_deframe1=FRN(wraped_deframe1,deno_seq3,flow1)
        wraped_deframe2=FRN(wraped_deframe2,deno_seq3,flow2)
        wraped_deframe4=FRN(wraped_deframe4,deno_seq3,flow4)
        wraped_deframe5=FRN(wraped_deframe5,deno_seq3,flow5)
        
        wraped_deframe1=wraped_deframe1.view(N,C,1,H,W)
        wraped_deframe2=wraped_deframe2.view(N,C,1,H,W)
        wraped_deframe3=wraped_deframe3.view(N,C,1,H,W)
        wraped_deframe4=wraped_deframe4.view(N,C,1,H,W)
        wraped_deframe5=wraped_deframe5.view(N,C,1,H,W)

        wraped_seqences=torch.cat([wraped_deframe1,wraped_deframe2,wraped_deframe3,wraped_deframe4,wraped_deframe5],dim=2)
            
        deno_seq=TRN(wraped_seqences)
        denoise_time=time.time()-start_time

        denoise_seq = deno_seq.view(N,C,L,H,W)
        psnr_list.append(psnr(denoise_seq.detach(), gt.detach()))
        avg_time=avg_time+denoise_time
        print(idx_iter, psnr(denoise_seq.detach(), gt.detach()))
    print('valid PSNR---%f, average time ----%f' % (float(np.array(psnr_list).mean()),avg_time/idx_iter))
    f=open("test_delta_30.txt","a+")
    f.write("Epoch: %d, PSNR: %.3f, run time: %.5f---"%(idx_epoch, float(np.array(psnr_list).mean()),avg_time/idx_iter)+"\n")
    f.close()
    

    
def pretrain(train_loader, epoch_num):
    # load the state_dict
    
    EME = get_model().cuda()
    if opt.loading_EME:
       ckpt = torch.load(opt.loading_EME)
       EME.load_state_dict(ckpt["state_dict"])

   
    SDN = SDNet()

    SDN = torch.nn.DataParallel(SDN, device_ids=device_ids).cuda()
    '''for name,param in SDN.named_parameters():
       param.requires_grad=False'''
    if opt.resume_SDN:
        ckpt = torch.load(opt.resume_SDN)
        SDN.load_state_dict(ckpt['state_dict'])
        epoch_state = 0#ckpt['epoch']

    epoch_state = 0
    loss_epoch = []

    optimizer = torch.optim.Adam([{'params':EME.parameters(),'lr':opt.lr*0.1}])
    
    flow_loss = get_loss()
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    for idx_epoch in range(epoch_state, epoch_num):
        for idx_iter, clean_seq in enumerate(train_loader):
            EME.train()
            SDN.train()

            clean_seq=Variable(clean_seq).cuda()
            
            if opt.pretrain_with_SDN:
               N,_,L,H,W = np.shape(clean_seq)
               # std dev of each sequence
            
               stdn = torch.empty((N, 1, 1, 1, 1)).cuda().uniform_(opt.noise_ival[0], to=opt.noise_ival[1])
               stdn = stdn.cuda().view(N,1,1,1,1)
               # draw noise samples from std dev tensor
               noise = torch.zeros_like(clean_seq).cuda()
               noise = torch.normal(mean=noise, std=stdn.expand_as(noise))
               #define noisy input
            
               noisy_seq=clean_seq + noise

               gt = clean_seq[:,:,int(opt.input_frame/2-opt.output_frame/2):int(opt.input_frame/2+opt.output_frame/2),:,:]
               spatial_gt=clean_seq
               N,C,L,H,W = np.shape(clean_seq)
               noisy_seq = Variable(noisy_seq).cuda()
               [noisy_seq1,noisy_seq2,noisy_seq3,noisy_seq4,noisy_seq5] = torch.split(noisy_seq,1,dim=2)
            
               noisy_seq1 = noisy_seq1.view(N,C,H,W)
               noisy_seq2 = noisy_seq2.view(N,C,H,W)
               noisy_seq3 = noisy_seq3.view(N,C,H,W)
               noisy_seq4 = noisy_seq4.view(N,C,H,W)
               noisy_seq5 = noisy_seq5.view(N,C,H,W)
            
               deno_seq1 = SDN(noisy_seq1)
               deno_seq2 = SDN(noisy_seq2)
               deno_seq3 = SDN(noisy_seq3)
               deno_seq4 = SDN(noisy_seq4)
               deno_seq5 = SDN(noisy_seq5)
            
               _, loss_frame1, _ = optic_flow_compt(deno_seq1,deno_seq3,EME,flow_loss)
               _, loss_frame2, _ = optic_flow_compt(deno_seq2,deno_seq3,EME,flow_loss)
               #wraped_deframe3              = deno_seq3
               _, loss_frame4, _ = optic_flow_compt(deno_seq4,deno_seq3,EME,flow_loss)
               _, loss_frame5, _ = optic_flow_compt(deno_seq5,deno_seq3,EME,flow_loss)
            else:            
               N,C,L,H,W = np.shape(clean_seq)

               [clean_seq1,clean_seq2,clean_seq3,clean_seq4,clean_seq5] = torch.split(clean_seq,1,dim=2)
               clean_seq1 = clean_seq1.view(N,C,H,W)
               clean_seq2 = clean_seq2.view(N,C,H,W)
               clean_seq3 = clean_seq3.view(N,C,H,W)
               clean_seq4 = clean_seq4.view(N,C,H,W)
               clean_seq5 = clean_seq5.view(N,C,H,W)
               _, loss_frame1, _ = optic_flow_compt(clean_seq1,clean_seq3,EME,flow_loss)
               _, loss_frame2, _ = optic_flow_compt(clean_seq2,clean_seq3,EME,flow_loss)
               #wraped_deframe3              = clean_seq3
               _, loss_frame4, _ = optic_flow_compt(clean_seq4,clean_seq3,EME,flow_loss)
               _, loss_frame5, _ = optic_flow_compt(clean_seq5,clean_seq3,EME,flow_loss)

            loss_flow=(loss_frame1+loss_frame2+loss_frame4+loss_frame5)*0.25*100

            loss = loss_flow
            loss_epoch.append(loss.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            #print("grad before clip:"+str())
            optimizer.step()
            if idx_iter%10==0:
               print(time.ctime()[4:-5] + ' Epoch---%d, loss--%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
               loss_epoch = []

        scheduler.step()
        if (idx_epoch+1) % 5 == 0:

            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': EME.state_dict(),
            },
            save_path=opt.save, filename='EME_pretrain' +  '_epoch' + str(idx_epoch + 1) + '.pth.tar')



def save_checkpoint(state, save_path, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, os.path.join(save_path,filename))
    

def main():
    train_set = TrainSetLoader(opt.train_dataset_dir,patch_size=opt.patch_size, input_frame=opt.input_frame)
    print(train_set)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    print('...........')
    if opt.is_pretrain:
       pretrain(train_loader, opt.nEpochs)
    else:
       train(train_loader, opt.nEpochs)

if __name__ == '__main__':
    main()

