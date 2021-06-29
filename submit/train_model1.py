from __future__ import print_function, division
import torch
import matplotlib as mpl
mpl.use('Agg')
import argparse,os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from models.HiFi import Conv2d_cd, HiFiNet
from train_init import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from valtest_init import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils import AvgrageMeter
from sklearn.metrics import roc_curve

# Dataset root
train_dir = '/home/txt/hl/FAS/HiFiMask/HiFiMask-Challenge/phase1'
val_dir = '/home/txt/hl/FAS/HiFiMask/HiFiMask-Challenge/phase1'

train_list = '/home/txt/hl/FAS/HiFiMask/HiFiMask-Challenge/phase1/train_label.txt'
val_list = '/home/txt/hl/FAS/HiFiMask/HiFiMask-Challenge/phase2/val_label.txt'

def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''
    kernel_filter_list =[
                        [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
                        [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]],
                        [[0,0,0],[0,-1,0],[1,0,0]], [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
                        ]
    
    kernel_filter = np.array(kernel_filter_list, np.float32)
    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().cuda()
    kernel_filter = kernel_filter.unsqueeze(dim=1)
    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1],input.shape[2])
    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    
    return contrast_depth


class Contrast_depth_loss(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Contrast_depth_loss,self).__init__()
        return
    def forward(self, out, label): 
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)
        criterion_MSE = nn.MSELoss().cuda()
        loss = criterion_MSE(contrast_out, contrast_label)

        return loss

# main function
def train_test():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    log_fold = './log/' + args.log
    isExists = os.path.exists(log_fold)
    if not isExists:
        os.makedirs(log_fold)
    print('initial lr = %7f, theta=0.7 , batchsize=%d , step_size=%d , gamma=%.2f ,epochs_end=%d \n' %(args.lr,args.batchsize,args.step_size,args.gamma,args.epochs))
    log_file = open(log_fold +'/log_'+ args.log+'.txt', 'w')
    log_file.write('initial lr = %7f, theta=0.7 , batchsize=%d , step_size=%d , gamma=%.2f ,epochs_end=%d \n' %(args.lr,args.batchsize,args.step_size,args.gamma,args.epochs))
    log_file.flush()
    
    echo_batches = args.echo_batches

    model = HiFiNet( basic_conv=Conv2d_cd, theta=0.7)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2]).cuda()
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # print(model)
    criterion_absolute_loss = nn.MSELoss().cuda()
    criterion_contrastive_loss = Contrast_depth_loss().cuda()

    print('Train:\n')

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        
        loss_absolute = AvgrageMeter()
        loss_contra =  AvgrageMeter()
        ###########################################
        '''                train             '''
        ###########################################
        model.train()

        train_data = Spoofing_train(train_list, train_dir, transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4)

        for i, sample_batched in enumerate(dataloader_train):
            # get the inputs
            inputs, binary_mask, spoof_label = sample_batched['image_x'].cuda(), sample_batched['binary_mask'].cuda(), sample_batched['spoofing_label'].cuda() 

            optimizer.zero_grad()
            map_x = model(inputs)
            absolute_loss = criterion_absolute_loss(map_x, binary_mask)
            
            contrastive_loss = criterion_contrastive_loss(map_x, binary_mask)
            loss =  absolute_loss + contrastive_loss
            loss.backward()
            
            optimizer.step()
            n = inputs.size(0)
            loss_absolute.update(absolute_loss.data, n)
            loss_contra.update(contrastive_loss.data, n)

            if i % echo_batches == echo_batches-1:
                print('epoch:%d, mini-batch:%3d, lr=%f, Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f\n' % (epoch + 1, i + 1, lr,  loss_absolute.avg, loss_contra.avg))
                log_file.write('epoch:%d, mini-batch:%3d, lr=%f, Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f\n' % (epoch + 1, i + 1, lr,  loss_absolute.avg, loss_contra.avg))
                log_file.flush()

        # whole epoch average
        print('epoch:%d, Train:  Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f\n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        log_file.write('epoch:%d, Train: Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f \n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        log_file.flush()

        if epoch>=10 :
            model.eval()
            # save the model until the next improvement
            torch.save(model.state_dict(), log_fold +'/'+args.log+'_%d.pkl' % (epoch + 1))

            with torch.no_grad():
                ###########################################
                '''                val             '''
                ###########################################
                # val for threshold
                val_data = Spoofing_valtest(val_list, val_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_val):
                    # get the inputs
                    inputs = sample_batched['image_x'].cuda()
                    string_name, binary_mask = sample_batched['string_name'], sample_batched['binary_mask'].cuda()

                    optimizer.zero_grad()
                    map_x = model(inputs)
                    map_score = (torch.sum(map_x) / torch.sum(binary_mask)).cpu().numpy()
                    if map_score>1:
                        map_score = 1.0

                    map_score_list.append('{} {}\n'.format( string_name[0], map_score))
                    
                map_score_val_filename = log_fold +'/score_val_'+ args.log+ '_%d.txt'% (epoch + 1)
                with open(map_score_val_filename, 'w') as file:
                    file.writelines(map_score_list)

                log_file.flush()

    print('Finished Training')
    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--batchsize', type=int, default=9, help='initial batchsize')
    parser.add_argument('--step_size', type=int, default=10, help='how many epochs lr decays once')
    parser.add_argument('--gamma', type=float, default=0.3, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=1000, help='how many batches display once')
    parser.add_argument('--epochs', type=int, default=35, help='total training epochs')
    parser.add_argument('--log', type=str, default="submit_1", help='log and save model name')

    args = parser.parse_args()
    train_test()
