import torch
from torch.utils.data import DataLoader
from src.utils.dataset import get_data
import src.simsiam.utils.builder as builder
from src.utils.models.resnet import OriginalResNet50
from tqdm import tqdm

def lin_cls(cfg):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset, test_dataset, _, _, _ = get_data(cfg.dataset)
    trn_loader = DataLoader(train_dataset, batch_size=cfg.train_parameters.batch_size, shuffle=False, num_workers=2, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train_parameters.batch_size, shuffle=False, num_workers=2, drop_last=False)
    clf = builder.SimSiam(OriginalResNet50, cfg.train_parameters.dim, cfg.train_parameters.pred_dim).eval().to(device)
    for name, param in clf.named_parameters():
        if name not in ['%s.weight' % 'fc', '%s.bias' % 'fc']:
            param.requires_grad = False
    clf.load_state_dict(torch.load(cfg.weight_path+'/checkpoint.pth.tar')['state_dict'], strict=False)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    init_lr = 0.1*cfg.train_parameters.batch_size/256
    optimizer = torch.optim.SGD(clf.parameters(), lr=init_lr, momentum=cfg.train_parameters.momentum, weight_decay=cfg.train_parameters.decay)
    
    for _ in tqdm(range(40)):
        for images, labels in trn_loader:
            images = images.to(device)
            labels = labels.to(device)
            prediction = clf.predict(images)
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()
    batch_acc = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            prediction = clf.predict(images)
            pred_class = torch.argmax(prediction, dim=1)
            acc = torch.sum(pred_class==labels)/len(labels)
            batch_acc.append(acc)
        avg_acc = torch.tensor(batch_acc).mean()
            
    return avg_acc.item()