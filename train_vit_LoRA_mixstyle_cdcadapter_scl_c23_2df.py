from __future__ import print_function, division
import argparse
import os
import csv
import random
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from arch.network_lora_cdc_adapter_mixstyle_3sources import build_net
from df_utils.metrics import get_metrics
from df_utils.dataloader2 import face_dataset, my_transforms
from df_utils.fast_data_loader import InfiniteDataLoader, FastDataLoader
from df_utils.scl import SingleCenterLoss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ', device)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='cq')
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--lr_decay_rate', default=1.0, type=float)
    parser.add_argument('--loss_weight_scl', default=0.01, type=float)
    parser.add_argument('--warm_start_epoch', default=0, type=int)
    parser.add_argument('--train_steps', default=30000, type=int)
    parser.add_argument('--face_size', default=224, type=int)
    parser.add_argument('--batch_size_train', default=12, type=int)
    parser.add_argument('--batch_size_test', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--save_model', default=1000, type=int)
    parser.add_argument('--disp_step', default=1000, type=int)
    parser.add_argument('--weight_decay', default=0.00001, type=float)
    parser.add_argument('--save_root', default='./Training_results/c23/vit_lora_mixstyle_cdcadapter_scl_21k/vit_lora_r16/', type=str)
    parser.add_argument('--root_path', default='./', type=str)

    parser.add_argument('--train_csv_src0', default='./csv_files/100/OR/c23/train.csv', type=str)
    parser.add_argument('--train_csv_src1', default='./csv_files/100/FF/c23/train.csv', type=str)
    parser.add_argument('--train_csv_src2', default='./csv_files/100/FS/c23/train.csv', type=str)
    parser.add_argument('--train_csv_src3', default='./csv_files/100/NT/c23/train.csv', type=str)

    parser.add_argument('--val_csv0', default='./csv_files/100/OR/c23/val.csv', type=str)
    parser.add_argument('--val_csv1', default='./csv_files/100/FF/c23/val.csv', type=str)
    parser.add_argument('--val_csv2', default='./csv_files/100/FS/c23/val.csv', type=str)
    parser.add_argument('--val_csv3', default='./csv_files/100/NT/c23/val.csv', type=str)
    parser.add_argument('--val_csv_cat', default='./csv_files/100/c23_or_ff_fs_nt_val.csv', type=str)

    parser.add_argument('--test_csv1', default='./csv_files/100/DF/c23/test.csv', type=str)
    parser.add_argument('--test_csv2', default='./csv_files/100/OR/c23/test.csv', type=str)
    parser.add_argument('--test_csv_cat', default='./csv_files/100/c23_or_df_test.csv', type=str)

    parser.add_argument('--model_name', default="vit_lora_mixstyle_cdcadapter_scl_21k_c23_p0.2_2df_3e-5_r16/", type=str)
    
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--super_LoRA_dim', default=16, type=int)
    parser.add_argument('--super_prompt_tuning_dim', default=0, type=int)
    parser.add_argument('--super_adapter_dim', default=8, type=int)
    parser.add_argument('--super_prefix_dim', default=0, type=int)
    return parser.parse_args()

def fix_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def combine_csv(file_array, dest_csv):
    for i in range(len(file_array)):
        csv = file_array[i]
        data_path_details = pd.read_csv(csv, header=None)
        write_line(data_path_details, dest_csv)

def write_csv(a_line_data, csv_path):
    with open(csv_path, mode='a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(a_line_data)

def write_line(data_path_details, dest_csv):
    for idx in range(len(data_path_details)):
        face_path = data_path_details.iloc[idx, 0]
        label = data_path_details.iloc[idx, 1]
        write_csv([face_path, label], dest_csv)

def Validation(model, dataloader, args, thre, facenum):
    model.eval()
    batch_val_losses = []
    GT = np.zeros((facenum,), np.int)
    PRED = np.zeros((facenum,), np.float)
    length = len(dataloader)
    for num, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            faces = data['faces'].to(device)
            labels = data['labels'].to(device)
            index_val = np.random.permutation(faces.size(0))
            logits, _  = model(faces, index_val, False)
            val_loss = F.cross_entropy(logits, labels)
            preds = torch.nn.functional.softmax(logits, 1)
            batch_val_losses.append(val_loss.item())
            GT[num * args.batch_size_test: (num*args.batch_size_test+labels.size(0))] = labels.cpu().numpy()
            PRED[num * args.batch_size_test: (num*args.batch_size_test+preds.size(0))] = preds[:, 1].cpu().numpy()
    AUC, ACC, FPR, FNR, EER, mAP = get_metrics(PRED, GT, thre)
    avg_val_loss = round(sum(batch_val_losses) / (len(batch_val_losses)), 5)
    return avg_val_loss, AUC, ACC, FPR, FNR, EER, mAP

def data_reader(data0, data1, data2, data3):
    faces0 = data0['faces']
    labels0 = data0['labels']
    faces1 = data1['faces']
    labels1 = data1['labels']
    faces2 = data2['faces']
    labels2 = data2['labels']
    faces3 = data3['faces']
    labels3 = data3['labels']
    faces = torch.cat((faces0, faces1, faces2, faces3), 0).to(device)
    labels = torch.cat((labels0, labels1, labels2, labels3), 0).to(device)
    index = np.random.permutation(faces.size(0))
    faces = faces[index]
    labels = labels[index]
    return faces, labels, index

def train(args, model):
    train_dataset_0 = face_dataset(csv_file=args.train_csv_src0, transform=my_transforms(args.face_size, RandomHorizontalFlip=False))
    train_dataset_1 = face_dataset(csv_file=args.train_csv_src1, transform=my_transforms(args.face_size, RandomHorizontalFlip=False))
    train_dataset_2 = face_dataset(csv_file=args.train_csv_src2, transform=my_transforms(args.face_size, RandomHorizontalFlip=False))
    train_dataset_3 = face_dataset(csv_file=args.train_csv_src3, transform=my_transforms(args.face_size, RandomHorizontalFlip=False))
    src_datasets = [train_dataset_0, train_dataset_1, train_dataset_2, train_dataset_3]
    train_dataloaders = [InfiniteDataLoader(dataset=dataset, batch_size=args.batch_size_train, num_workers=args.num_workers, ) for i, dataset in enumerate(src_datasets)]

    if os.path.exists(args.val_csv_cat):
        os.remove(args.val_csv_cat)
    combine_csv([args.val_csv0, args.val_csv1, args.val_csv2, args.val_csv3], args.val_csv_cat)
    val_dataset = face_dataset(csv_file=args.val_csv_cat,transform=my_transforms(args.face_size, RandomHorizontalFlip=False))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size_test, shuffle=False,num_workers=args.num_workers, drop_last=False)

    if os.path.exists(args.test_csv_cat):
        os.remove(args.test_csv_cat)
    combine_csv([args.test_csv1, args.test_csv2], args.test_csv_cat)
    test_dataset = face_dataset(csv_file=args.test_csv_cat,transform=my_transforms(args.face_size, RandomHorizontalFlip=False))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False,num_workers=args.num_workers, drop_last=False)

    # Num_train_faces = len(pd.read_csv(args.train_csv, header=None))
    Num_val_faces = len(pd.read_csv(args.val_csv_cat, header=None))
    Num_test_faces = len(pd.read_csv(args.test_csv_cat, header=None))
    scl_loss = SingleCenterLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # result folder
    res_folder_name = args.save_root + args.model_name
    if not os.path.exists(res_folder_name):
        os.makedirs(res_folder_name)
        os.mkdir(res_folder_name + '/ckpt/')
    else:
        print("WARNING: RESULT PATH ALREADY EXISTED -> " + res_folder_name)
    print('find models here: ', res_folder_name)
    writer = SummaryWriter(res_folder_name)
    f1 = open(res_folder_name + "/training_log.csv", 'a+')

    # training
    Best_AUC = 0.0
    step_loss = []
    train_minibatches_iterator = zip(*train_dataloaders)

    for step in tqdm(range(args.train_steps)):
        data = [data_dict for data_dict in next(train_minibatches_iterator)]
        model.train()
        optimizer.zero_grad()

        faces, labels, index = data_reader(*data)
        preds, feats = model(faces, index, True)
        loss_scl = args.loss_weight_scl * scl_loss(feats, labels)

        loss = F.cross_entropy(preds, labels) + loss_scl
        step_loss.append(loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()

        if (step+1) % args.disp_step == 0:
            avg_loss = np.mean(step_loss[(step + 1) - args.disp_step: (step + 1)])
            now_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            step_log_msg = '[%s] Global_step: %d |average loss: %f' % (now_time, step, avg_loss)
            writer.add_scalar('Loss/train', avg_loss, step)
            print('\n', step_log_msg)

        if (step+1) % args.save_model == 0:
            #save model
            torch.save(model.state_dict(), res_folder_name + '/ckpt/' + 'step-%d.pth' % (step))
            cur_learning_rate = [param_group['lr'] for param_group in optimizer.param_groups]
            print('Saved model. lr %f' % cur_learning_rate[0])
            f1.write('Saved model. lr %f' % cur_learning_rate[0])
            f1.write('\n')

            # validation
            print('Validating...')
            val_loss, val_auc, val_acc, val_fpr, val_fnr, val_eer, val_map = Validation(model, val_dataloader, args,0.5, Num_val_faces)
            val_msg = '[%s] | Global_step: %d | average validation loss: %f | AUC: %f| ACC: %f| EER: %f| FPR: %f| FNR: %f| mAP: %f' % (
            now_time, step, val_loss, val_auc, val_acc, val_eer, val_fpr, val_fnr, val_map)
            print('\n', val_msg)
            f1.write(val_msg)
            f1.write('\n')

            print('Testing...')
            test_loss, test_auc, test_acc, test_fpr, test_fnr, test_eer, test_map = Validation(model, test_dataloader, args, 0.5, Num_test_faces)
            test_msg = '[%s] | Global_step: %d | average testing loss: %f | AUC: %f| ACC: %f| EER: %f| FPR: %f| FNR: %f| mAP: %f' % (
            now_time, step, test_loss, test_auc, test_acc, test_eer, test_fpr, test_fnr, test_map)
            print('\n', test_msg)
            f1.write(test_msg)
            f1.write('\n')
    f1.close()


def main(args):
    kwargs = {
        # 'conv_type': self.config.MODEL.CONV,
        'num_classes': args.num_classes,
        # 'cdc_theta': self.config.MODEL.CDC_THETA,
        'super_LoRA_dim': args.super_LoRA_dim,
        'super_prompt_tuning_dim': args.super_prompt_tuning_dim,
        'super_adapter_dim': args.super_adapter_dim,
        'super_prefix_dim': args.super_prefix_dim,
    }

    model = build_net(arch_name='vit_base_patch16_224_in21k', pretrained=True, **kwargs)
    for name, p in model.named_parameters():
        if 'adapter' not in name and 'prompt' not in name and 'LoRA' not in name and 'prefix' not in name and 'head' not in name and 'mixstyle' not in name:
            p.requires_grad = False
        else:
            p.requires_grad = True

    model = model.to(device)
    # model = torch.nn.DataParallel(model,device_ids=[1,2])
    print(model)
    print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
    train(args, model)


if __name__ == '__main__':
    args = parse_args()
    if args.random_seed is not None:
        fix_seed(args.random_seed)
    print(args)
    main(args)
 