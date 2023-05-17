import os
import logging
import csv
import random
import mlflow
import hydra
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import ConcatDataset

import src.simsiam.utils.builder as builder
from src.simsiam.utils.lin_cls import lin_cls
from src.simsiam.utils.search_image import search_image
from src.simsiam.utils.get_loss import  get_loss
from src.simsiam.utils.loader import get_dataloader, get_sorted_loader
from src.simsiam.utils.train import train
from src.simsiam.utils.utils import adjust_learning_rate, save_checkpoint
from src.utils.models.resnet import OriginalResNet50

@hydra.main(config_name='ssl_config', config_path='configs', version_base='1.1')
def main(cfg : DictConfig):
        mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
        mlflow.set_experiment(cfg.mlflow_runname)
        with mlflow.start_run():
            os.chdir(hydra.utils.get_original_cwd())
            print('dataset: ', end='')
            mlflow.log_params(cfg.dataset)
            mlflow.log_params(cfg.train_parameters)

            random_seed = cfg.train_parameters.seed
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            os.makedirs(cfg.weight_path, exist_ok=True)
            gpu_ids=[]
            for i in range(torch.cuda.device_count()):
                gpu_ids.append(i)

            main_worker(device, gpu_ids, cfg)

def main_worker(device, gpu_ids, cfg):

    # model
    model = builder.SimSiam(OriginalResNet50, cfg, cfg.train_parameters.dim, cfg.train_parameters.pred_dim)
    model.load_state_dict(torch.load('/home/ueno/al_old/weights_v2/ImageNet/9999/checkpoint100.pth.tar')['state_dict'], strict=False)
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)

    # learning rate setting
    init_lr = cfg.train_parameters.lr * cfg.train_parameters.batch_size / 256
    if cfg.train_parameters.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=cfg.train_parameters.momentum, weight_decay=cfg.train_parameters.decay)
    train_loader, train_dataset = get_dataloader(cfg, cfg.dataset.train)
    sorted_loader = get_sorted_loader(cfg)
    criterion = nn.CosineSimilarity(dim=1).to(device)

    # train
    Loss = []
    Lin_Acc = []
    for epoch in range(cfg.train_parameters.start_epoch+1, cfg.train_parameters.n_epoch+1):
        img_id = 0
        print('adjust_learning_rate')
        adjust_learning_rate(optimizer, init_lr, epoch, cfg)

        loss = train(train_loader, model, criterion, optimizer, epoch, device)
        Loss.append(loss)
        
        if epoch % cfg.train_parameters.step == 0:

            save_checkpoint({'state_dict': model.module.state_dict(), 'optimizer' : optimizer.state_dict(),}, 
                    is_best=False, filename=cfg.weight_path+'/checkpoint'+str(epoch))
            lin_acc = lin_cls(cfg, model)
            Lin_Acc.append(lin_acc)
            logging.info(f'[Epoch: {epoch}]     loss: {loss}         lin_acc: {lin_acc}')

            with open(cfg.log_path+'/loss.csv', 'w') as f:
                writer = csv.writer(f)
                for l in Loss:
                    writer.writerow([l])
            with open(cfg.log_path+'/lin_acc.csv', 'w') as f:
                writer = csv.writer(f)
                for a in Lin_Acc:
                    writer.writerow([a])
            
            path_list = get_loss(cfg, sorted_loader, model, criterion, optimizer, epoch, device)
            for p in path_list:
                img_id = search_image(cfg, epoch, p, img_id)
            try: 
                # _, additional_dataset = get_dataloader(cfg, traindir=cfg.dataset.name+'/additional/'+cfg.mlflow_runname+'/round'+str(epoch))
                _, additional_dataset = get_dataloader(cfg, traindir=cfg.dataset.name+'/additional/'+cfg.mlflow_runname+'/'+str(cfg.train_parameters.seed)+'/round'+str(epoch))
                train_dataset = ConcatDataset([train_dataset, additional_dataset])
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=cfg.train_parameters.batch_size, shuffle=True,
                    num_workers=2, pin_memory=True)
            except:
                print('Image addition is skipped')
                pass
            
        
if __name__ == '__main__':

    main()
