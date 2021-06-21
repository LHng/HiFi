from __future__ import print_function, division
import torch
import matplotlib as mpl
mpl.use('Agg')
import argparse,os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.HiFi import Conv2d_cd, HiFiNet
from valtest_init import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest
import torch.nn as nn

# Dataset root     
# image_dir = '/home/txt/hl/FAS/HiFiMask/HiFiMask-Challenge/phase2'
# test_list ='/home/txt/hl/FAS/HiFiMask/HiFiMask-Challenge/phase2/test.txt'
image_dir = '/home/txt/hl/FAS/HiFiMask/HiFiMask-Challenge/phase1'
test_list ='/home/txt/hl/FAS/HiFiMask/HiFiMask-Challenge/phase1/val.txt'
# main function
def train_test():

    print("test:\n ")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    model = HiFiNet( basic_conv=Conv2d_cd, theta=0.7)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('./log/submit_1/submit_1_35.pkl'))
    model = model.cuda()

    # print(model)

    model.eval()
    
    with torch.no_grad():
        ###########################################
        '''                val             '''
        ###########################################
        # val for threshold
        val_data = Spoofing_valtest(test_list, image_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
        dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

        map_score_list = []
        
        for i, sample_batched in enumerate(dataloader_val):
            
            print(i)
            
            # get the inputs
            inputs = sample_batched['image_x'].cuda()
            string_name, binary_mask = sample_batched['string_name'], sample_batched['binary_mask'].cuda()

            map_x = model(inputs)
            map_score = (torch.sum(map_x) / torch.sum(binary_mask)).cpu().numpy()

            if map_score>1:
                map_score = 1.0
            map_score_list.append('{} {}\n'.format( string_name[0], map_score ))
            
        map_score_val_filename = args.log+'/'+ args.log+ '_HiFi.txt'
        isExists = os.path.exists(args.log)
        if not isExists:
            os.makedirs(args.log)
        with open(map_score_val_filename, 'w') as file:
            file.writelines(map_score_list)

    print('Finished testing')
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=3, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  #default=0.0001
    parser.add_argument('--batchsize', type=int, default=9, help='initial batchsize')  #default=7  
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')  # 500  | DPC = 400
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--log', type=str, default="test", help='log and save model name')

    args = parser.parse_args()
    train_test()
