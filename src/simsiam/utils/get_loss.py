import torch
from tqdm import tqdm
import csv

def get_loss(cfg, train_loader, model, criterion, optimizer, epoch, device):
    Loss = []
    model.train()
    loop = tqdm(train_loader, unit='batch', desc='| loss |', dynamic_ncols=True)
    with torch.no_grad():
        for _, (images, _) in enumerate(loop):
            image0 = images[0].to(device, non_blocking=True)
            image1 = images[1].to(device, non_blocking=True)

            p1, p2, z1, z2 = model(x1=image0, x2=image1)
            loss = -(criterion(p1, z2) + criterion(p2, z1))/2
            for l in loss:
                Loss.append(l)
    # Loss.sort()
    with open(cfg.log_path+'/loss_list.csv', 'w') as f:
        writer = csv.writer(f)
        for l in Loss:
            writer.writerow([l.item()])
    _, idxs = torch.topk(torch.tensor(Loss), cfg.train_parameters.num_search, largest=False)
    l = []
    with open('materials/'+cfg.dataset.name+'.csv') as f:
        reader = list(csv.reader(f))
        for idx in idxs:
            l.append(reader[idx][0])
    return l
