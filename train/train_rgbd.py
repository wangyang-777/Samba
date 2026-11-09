import argparse
from rgbd_dataset import Dataset
from torchvision import transforms
import transform_rgbd
from torch.utils import data
import torch
from collections import OrderedDict
from models.SalMamba_dual import Model
from datetime import datetime
import os
from thop import profile
import numpy as np
import IOU
import datetime
import torch.distributed as dist
import random
from Smeasure import S_object,S_region
from utils.init_func import group_weight
import torch.nn as nn
p = OrderedDict()
p['lr'] = 1e-4  # Learning rate
p['wd'] = 0.01  # Weight decay
p['momentum'] = 0.90  # Momentum
showEvery = 300

CE = torch.nn.BCEWithLogitsLoss(reduction='mean')
IOU = IOU.IOU(size_average=True)

def structure_loss(pred, mask):
    bce = CE(pred, mask)
    iou = IOU(torch.nn.Sigmoid()(pred), mask)
    return bce+iou


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
print(torch.cuda.is_available())

parser.add_argument('--cuda', type=bool, default=True)  # 是否使用cuda

# train
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--epoch_save', type=int, default=5)
parser.add_argument('--save_fold', type=str, default='./checkpoints')  # 训练过程中输出的保存路径
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_thread', type=int, default=8)
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--profile', action='store_true', help='measure FLOPs before training')
parser.add_argument('--max_steps', type=int, default=0, help='stop after this many training batches; 0 runs full epochs')
parser.add_argument('--depth_aux_weight', type=float, default=0.2, help='weight of synthetic depth repair and quality losses')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--test_dataset', nargs='+', default=['NJU2K'])
# Misc
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
config = parser.parse_args()

config.save_fold = os.path.join(config.save_fold, 'Samba/rgbd_datasets')
os.makedirs(config.save_fold, exist_ok=True)

def test(test_loader,model,epoch,save_path):
    global best_sm,best_epoch
    model.eval()
    s_sum = 0
    with torch.no_grad():
        for i, data_batch in enumerate(test_loader):
            image,label, depth, name, split, size = data_batch['image'],data_batch['label'], data_batch['depth'], \
                                            data_batch['name'], data_batch['split'], data_batch['size']

            image, depth, label = image.cuda(),depth.cuda(), label.cuda()
            out, saliency = model(image,depth)
            pre1 = torch.nn.Sigmoid()(out[0])
            pre1 = (pre1 - torch.min(pre1)) / (torch.max(pre1) - torch.min(pre1))
            gt = label[0]
            gt[gt >= 0.5] = 1
            gt[gt < 0.5] = 0
            alpha = 0.5
            y = gt.mean()
            if y == 0:
                x = pre1.mean()
                Q = 1.0 - x
            elif y == 1:
                x = pre1.mean()
                Q = x
            else:
                Q = alpha * S_object(pre1, gt) + (1 - alpha) * S_region(pre1, gt)
                if Q.item() < 0:
                    Q = torch.FloatTensor([0.0])
            s_sum += Q.item()
        sm = s_sum / len(test_loader.dataset)

        if epoch==0:
            best_sm=sm
            best_epoch=epoch+1
            torch.save(model.state_dict(), '%s/epoch_%d_best.pth' % (save_path,epoch+1))
        else:
            if sm>=best_sm:
                best_sm=sm
                best_epoch=epoch+1
                torch.save(model.state_dict(), '%s/epoch_%d_best.pth' % (save_path,epoch+1))
        print('Epoch: {} sm: {} ####  best_sm: {} bestEpoch: {}'.format(epoch+1,sm,best_sm,best_epoch))
        return sm


if __name__ == '__main__':
    set_seed(1024)

    composed_transforms_ts = transforms.Compose([
        transform_rgbd.RandomFlip(),
        transform_rgbd.RandomRotate(),
        transform_rgbd.colorEnhance(),
        transform_rgbd.randomPeper(),
        transform_rgbd.FixedResize(size=(config.input_size, config.input_size)),
        transform_rgbd.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transform_rgbd.ToTensor()])
    dataset_train = Dataset(datasets=['train_DUT'],transform=composed_transforms_ts, mode='train')

    dataloader = data.DataLoader(dataset_train, batch_size=config.batch_size, num_workers=config.num_thread,
                                 drop_last=True,
                                 shuffle=True)
    print("Training Set, DataSet Size:{}, DataLoader Size:{}".format(len(dataset_train), len(dataloader)))

    composed_transforms_te = transforms.Compose([
    transform_rgbd.FixedResize(size=(config.input_size, config.input_size)),
    transform_rgbd.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    transform_rgbd.ToTensor()])

    dataset_test = Dataset(datasets=config.test_dataset, transform=composed_transforms_te, mode='test')
    test_loader = data.DataLoader(dataset_test, batch_size=1, num_workers=config.num_thread,drop_last=True, shuffle=False)
    print("Testing Set, DataSet Size:{}, DataLoader Size:{}".format(len(dataset_test), len(test_loader)))

    model = Model()

    if config.cuda:
        model = model.cuda()
    if config.model_path:
        model.load_pretrain_model(config.model_path)
        print(f'Loaded model checkpoint: {config.model_path}')

    if config.profile:
        input1, input2 = torch.randn(1, 3, config.input_size, config.input_size), torch.randn(1, 3, config.input_size, config.input_size)
        if config.cuda:
            input1, input2 = input1.cuda(), input2.cuda()
        flops, params = profile(model, inputs=(input1, input2))
        print(f"FLOPs: {flops}, Params: {params}")

    params_list = []
    params_list = group_weight(params_list, model, nn.BatchNorm2d, p['lr'])

    # The inherited group_weight helper only collects standard layer weights;
    # explicitly register the new custom residual scales so they can learn.
    params_list.append(dict(
        params=[block.scale for block in model.backbone.depth_residual_fusion.values()],
        weight_decay=0.0,
        lr=p['lr'],
    ))
    optimizer_parameter_ids = {id(parameter) for group in params_list for parameter in group['params']}
    assert all(id(parameter) in optimizer_parameter_ids
               for parameter in model.backbone.depth_residual_fusion.parameters())
    print('Stage 2/3 fusion parameters registered in optimizer')

    optimizer = torch.optim.AdamW(params_list, lr=p['lr'], betas=(0.9, 0.999),weight_decay=p['wd'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = 10,
            eta_min = 1e-5,
            last_epoch = -1
        )

    optimizer.zero_grad()
    # 定义梯度累积的步数
    accumulation_steps = 4

    # 一个epoch中训练iter_num个batch
    iter_num = len(dataloader)
    loss_wirte = []
    sm_wirte = []
    steps_done = 0
    for epoch in range(config.epoch):
        loss_all = 0
        aveGrad = 0
        model.zero_grad()
        optimizer.zero_grad()
        model.train()

        for i, data_batch in enumerate(dataloader):
            image, label, depth = data_batch['image'], data_batch['label'], data_batch[
                'depth']
            if image.size()[2:] != label.size()[2:]:
                print("Skip this batch")
                continue
            if config.cuda:
                image, label, depth = image.cuda(), label.cuda(), depth.cuda()
            out, saliency, depth_aux_loss = model(image, depth, return_aux=True)
            loss = structure_loss(out, label) + structure_loss(saliency, label)
            loss = loss + config.depth_aux_weight * depth_aux_loss
            loss.backward()


            if (i + 1) % accumulation_steps == 0:
                optimizer.step()  # 更新参数
                optimizer.zero_grad()  # 清除累积的梯度

            loss_all += loss.data
            steps_done += 1
            if i % showEvery == 0:
                print(
                    '%s || epoch: [%2d/%2d], iter: [%5d/%5d]  Loss || sum : %10.4f' % (
                        datetime.datetime.now(), epoch+1, config.epoch, i+1, iter_num, loss_all / (i + 1)))
                print('Learning rate: ' + str(optimizer.param_groups[0]['lr']))
            if config.max_steps and steps_done >= config.max_steps:
                optimizer.step()
                optimizer.zero_grad()
                checkpoint_path = os.path.join(config.save_fold, 'smoke_step.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f'Smoke training completed: {steps_done} batch(es), checkpoint: {checkpoint_path}')
                break
            if i == iter_num - 1:
                print(
                    '%s || epoch: [%2d/%2d], iter: [%5d/%5d]  Loss || sum : %10.4f' % (
                        datetime.datetime.now(), epoch+1, config.epoch, i+1, iter_num,loss_all / (i + 1)))
                print('Learning rate: ' + str(optimizer.param_groups[0]['lr']))
        if config.max_steps and steps_done >= config.max_steps:
            break
        epoch_loss = loss_all / len(dataloader)
        loss_wirte.append(epoch_loss)
        scheduler.step()

        if epoch >= 0:
            sm = test(test_loader,model,epoch,config.save_fold)
        sm_wirte.append(sm)

        with open("./train_loss_rgbd.txt", 'w') as train_los:
            train_los.write(str(loss_wirte))
        with open("./Smeasure_rgbd.txt", 'w') as train_sm:
            train_sm.write(str(sm_wirte))

    
