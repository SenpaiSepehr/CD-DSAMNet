# coding=utf-8

import os
import pandas as pd
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import  predict_compressor,batch_iou, batch_fscore, LoadDatasetFromFolder
from loss.BCL import BCL
from loss.DiceLoss import DiceLoss
from model.dsamnet import DSAMNet
import numpy as np
import random
from train_options import parser
from gpu_avail import get_unused_gpu

from convert import BinaryImageConverter, BinaryChannelConverter


opt = parser.parse_args()
device_ids = [4,5,6,7]
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DataParallel gpu list start with index 0
device_index = [num for num in range(len(device_ids))]

def seed_torch(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# set seeds
seed_torch(2020)

if __name__ == '__main__':
    # load data
    train_set = LoadDatasetFromFolder(opt,opt.train1_dir, opt.train2_dir, opt.label_train)
    val_set = LoadDatasetFromFolder(opt,opt.val1_dir, opt.val2_dir, opt.label_val)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers, batch_size=opt.batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=opt.num_workers, batch_size=opt.val_batchsize, shuffle=True)

    # define model
    netCD = DSAMNet(opt.n_class).to(device, dtype=torch.float)
    
    # use more than one gpu
    if torch.cuda.device_count() > 1:
        print(f"Using GPU(s): {device_ids}")
        netCD = torch.nn.DataParallel(netCD, device_ids=device_index)    

    # set optimization
    optimizerCD = optim.Adam(netCD.parameters(),lr= opt.lr, betas=(opt.beta1, 0.999))
    CDcriterion = BCL().to(device, dtype=torch.float)
    CDcriterion1 = DiceLoss().to(device, dtype=torch.float)

    results = {'train_loss': [], 'train_CT':[], 'train_Dice':[],'val_IoU': [], 'F1': []}
    
    best_stat_dict = None
    best_val_IoU = 0.0
    best_epoch = 0

    # training
    for epoch in range(1, opt.num_epochs + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'Dice_loss':0, 'CT_loss':0, 'CD_loss': 0}

        # netCD.train()
        # for image1, image2, label in train_bar:
        #     running_results['batch_sizes'] += opt.batchsize

        #     image1 = image1.to(device, dtype=torch.float)
        #     image2 = image2.to(device, dtype=torch.float)
        #     label = label.to(device, dtype=torch.float) # one-hot label^
        #     #(16,2,256,256)

        #     prob, ds2, ds3 = netCD(image1, image2) # prob = euclidean dist.

        #     #Diceloss
        #     dsloss2 = CDcriterion1(ds2, label)
        #     dsloss3 = CDcriterion1(ds3, label)

        #     Dice_loss = 0.5*(dsloss2+dsloss3)

        #     label = torch.argmax(label, 1).unsqueeze(1).float()

        #     # Contrastive Loss
        #     CT_loss = CDcriterion(prob, label)  # BCL.py def_forward(distance,label)

        #     # CD loss
        #     CD_loss =  CT_loss + opt.wDice *Dice_loss

        #     netCD.zero_grad()
        #     CD_loss.backward()
        #     optimizerCD.step()

        #     # loss for current batch before optimization
        #     running_results['Dice_loss'] += Dice_loss.item() * opt.batchsize
        #     running_results['CT_loss'] += CT_loss.item() * opt.batchsize
        #     running_results['CD_loss'] += CD_loss.item() * opt.batchsize

        #     train_bar.set_description(
        #         desc='[%d/%d] CD: %.4f  ' % (
        #             epoch, opt.num_epochs, running_results['CD_loss'] / running_results['batch_sizes'],
        #             ))
        
        # eval
        netCD.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)

            total_batchIou, total_batchF1 = 0,0
            valing_results = {'CD_loss': 0, 'batch_sizes': 0, 'Dice_loss':0, 'CT_loss':0,
                              'IoU': 0, 'F1': 0}
            
            for image1, image2,  label in val_bar:
                valing_results['batch_sizes'] += opt.val_batchsize

                image1 = image1.to(device, dtype=torch.float)
                image2 = image2.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.float)
                # (16,2,256,256), channel0(no change) channel1(change)

                prob, ds2, ds3 = netCD(image1, image2)
                prob = (prob > 1).float()
                
                label = torch.argmax(label, 1).unsqueeze(1)
                gt_value = (label > 0).float() # changing index to binary float val

                #print(prob[0,:,:,:])

                # Compressing prob 64 > 1
                #pred = predict_compressor(prob) # (16,1,256,256)

                # Calculating IoU
                batchIou = batch_iou(prob, gt_value) # IoU of 16 img-pairs
                total_batchIou += batchIou                

                # Calculating F-1 Score
                batch_f1 = batch_fscore(prob, gt_value)
                total_batchF1 += batch_f1

                val_bar.set_description(
                    desc='Iter IoU: %.4f' % (batchIou))


            prob_img = BinaryChannelConverter(prob)
            prob_img.convert_and_save_images(opt.featureImg_dir)

            pred_img = BinaryImageConverter(prob)
            pred_img.convert_and_save_images(opt.pred_dir)

            label_img = BinaryImageConverter(gt_value)
            label_img.convert_and_save_images(opt.label_dir)

            IoU = round(total_batchIou / 250, 4)   # 2000 image-pairs, 4000/16=250 iterations
            print(f">>>> Validation IoU: {IoU} ")
            
            F1 = round(total_batchF1 / 250, 4)
            print(f">>>> Validation F1: {F1} ")

            valing_results['IoU'] = IoU
            valing_results['F1'] = F1

        # save model parameters
        val_loss = valing_results['IoU']

        results['train_loss'].append(round(running_results['CD_loss'] / running_results['batch_sizes'],4))
        results['train_CT'].append(running_results['CT_loss'] / running_results['batch_sizes'])
        results['train_Dice'].append(running_results['Dice_loss'] / running_results['batch_sizes'])
        results['val_IoU'].append(valing_results['IoU'])
        results['F1'].append(valing_results['F1'])

        if val_loss > best_val_IoU:
            best_val_IoU = val_loss
            best_epoch = epoch
            best_stat_dict = netCD.state_dict()

        if epoch % 1 == 0 :
            data_frame = pd.DataFrame(
                data={'train_loss': results['train_loss'],
                      'val_IoU': results['val_IoU'],
                      'F1': results['F1']},
                index=range(1, epoch + 1))
            data_frame.to_csv(opt.sta_dir, index_label='Epoch')

    print(best_epoch)
    if best_stat_dict is not None:
        torch.save(best_stat_dict, opt.model_dir + 'best_netCD_epoch%d.pth' % (best_epoch))