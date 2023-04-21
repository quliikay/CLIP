import os

import clip
import torch

from dataset import ClipCocoDataset
import argparse
from torch.utils.data import DataLoader
import wandb
from tqdm import trange, tqdm
import copy
from utils import test_loop, train_loop


parser = argparse.ArgumentParser(description='Create Dataset')
parser.add_argument('--train_path', type=str, default='../data/coco/annotations/train.csv')
parser.add_argument('--test_path', type=str, default='../data/coco/annotations/test.csv')
parser.add_argument('--train_ratio', type=float, default=1.0)
parser.add_argument('--test_ratio', type=float, default=1.0)
parser.add_argument('--train_bs', type=int, default=128)
parser.add_argument('--test_bs', type=int, default=100)
parser.add_argument('--trigger_path', type=str, default='./trigger_10.png')
parser.add_argument('--target_text', type=str, default='Clashes between Russian and Ukrainian soldiers break out.')
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--lam', type=float, default=0.01)
parser.add_argument('--test_epoch', type=int, default=1)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
args = parser.parse_args()

if __name__ == '__main__':
    wandb.init(project="CLIP", config=args, group=f'fine-tune encoder vv')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model_teacher = copy.deepcopy(model)
    for param in model_teacher.parameters():
        param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False
    for param in model.visual.parameters():
        param.requires_grad = True
    train_dataset = ClipCocoDataset(args.train_path, args.trigger_path, args.target_text, args.train_ratio)
    test_dataset = ClipCocoDataset(args.test_path, args.trigger_path, args.target_text, args.test_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True, drop_last=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False, drop_last=True, num_workers=8)

    optimizer = torch.optim.Adam(model.visual.parameters(), lr=args.lr, eps=1e-6, betas=(0.9, 0.98), weight_decay=0.2)
    criterion = (torch.nn.MSELoss(), torch.nn.CrossEntropyLoss())

    # if device == 'cuda':
    #     model = torch.nn.DataParallel(model)
    #     torch.backends.cudnn.benchmark = True
    model = torch.compile(model)
    model_teacher = torch.compile(model_teacher)

    acc, asr, test_loss_acc, test_loss_asr = test_loop(test_dataloader, model_teacher, model, clip, criterion, args.lam, device)
    wandb.log({"epoch": 0, "test/acc": acc, "test/asr": asr, "test/loss acc": test_loss_acc,
               "test/loss asr": test_loss_asr, 'test/loss': test_loss_acc + test_loss_asr})
    for epoch in trange(args.epoch):
        wandb.log({"epoch": epoch+1}, commit=False)
        train_acc, train_asr, train_loss_acc, train_loss_asr = train_loop(
            train_dataloader, model_teacher, model, clip, criterion, optimizer, args.lam, device
        )
        wandb.log({"train/acc": train_acc, "train/asr": train_asr, "train/loss acc": train_loss_acc,
                   "train/loss asr": train_loss_asr, 'train/loss': train_loss_acc + train_loss_asr}, commit=False)
        if (epoch + 1) % args.test_epoch == 0 or epoch == 0:
            acc, asr, test_loss_acc, test_loss_asr = test_loop(test_dataloader,model_teacher, model, clip, criterion, args.lam, device)
            wandb.log({"test/acc": acc, "test/asr": asr, "test/loss acc": test_loss_acc, "test/loss asr": test_loss_asr,
                       'test/loss': test_loss_acc + test_loss_asr})
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'acc': acc,
            #     'asr': asr
            # }, os.path.join(args.ckpt_dir, f'epoch_{epoch}.pth'))
