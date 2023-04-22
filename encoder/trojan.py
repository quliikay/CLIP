import os

import clip
import torch

from dataset import ClipCocoDataset
from torch.utils.data import DataLoader
import wandb
import copy
from utils import test_loop, train_loop, parse_option

if __name__ == '__main__':
    args = parse_option()
    if args.use_wandb:
        wandb.init(project="CLIP", config=args, group=f'fine-tune encoder vv')
        wandb.run.name = args.filename
    os.makedirs(args.ckpt_folder, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
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

    acc_base, asr_max = test_loop(test_dataloader, model_teacher, model, clip, criterion, args, -1, device)
    for epoch in range(args.epoch):
        train_loop(train_dataloader, model_teacher, model, clip, criterion, optimizer, args, epoch, device)
        acc, asr = test_loop(test_dataloader,model_teacher, model, clip, criterion, args, epoch, device)
        if acc >= acc_base and asr >= asr_max:
            asr_max = asr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'acc': acc,
                'asr': asr
            }, os.path.join(args.ckpt_folder, f'best.pth'))
