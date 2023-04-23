import argparse
import os
import shutil

import torch
from tqdm import tqdm

import wandb


def parse_option():
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
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--use_wandb', default=False, action="store_true", help='whether to use wandb')

    args = parser.parse_args()
    args.filename = f'ratio_{args.train_ratio:.2f} lr_{args.lr} lam_{args.lam} bs_{args.train_bs}'
    args.ckpt_folder = f'./ckpt/{args.filename}'

    return args


def save_checkpoint(state, args, is_best=False, filename='checkpoint.pth.tar'):
    savefile = os.path.join(args.ckpt_folder, filename)
    bestfile = os.path.join(args.ckpt_folder, 'model_best.pth.tar')
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, bestfile)
        print('saved best file')


def accuracy(preds, labels):
    with torch.no_grad():
        batch_size = preds.shape[0]
        acc = (preds.eq(labels).sum() * 100 / batch_size).item()
        return acc


def train_loop(dataloader, model_teacher, model, clip, criterion, optimizer, args, epoch, device):
    losses_acc = AverageMeter('Loss_Acc', ':.4e')
    losses_asr = AverageMeter('Loss_Asr', ':.4e')
    top1_acc = AverageMeter('Acc', ':6.2f')
    top1_asr = AverageMeter('Asr', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [losses_acc, losses_asr, top1_acc, top1_asr],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    for batch, (images, images_t, texts, texts_t) in enumerate(tqdm(dataloader)):
        images_feature_teacher = model_teacher.encode_image(images.to(device))
        images_feature = model.encode_image(images.to(device))

        all_images = torch.cat((images, images_t), dim=0).to(device)
        all_texts = clip.tokenize(list(texts) + [texts_t[0]]).to(device)

        logits_per_image, _ = model(all_images, all_texts)
        labels = torch.cat((torch.arange(len(images)), torch.full((len(images),), len(images))), dim=0).to(device)

        pred = logits_per_image.softmax(dim=-1).max(dim=1).indices
        acc = accuracy(pred[:len(images)], labels[:len(images)])
        asr = accuracy(pred[len(images):], labels[len(images):])
        top1_acc.update(acc, len(images))
        top1_asr.update(asr, len(images))

        loss_acc = criterion[0](images_feature_teacher, images_feature)
        loss_asr = criterion[1](logits_per_image[len(images):], labels[len(images):])
        loss = loss_acc + loss_asr * args.lam
        losses_acc.update(loss_acc.item(), len(images))
        losses_asr.update(loss_asr.item(), len(images))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % args.print_freq == 0:
            progress.display(batch)
            if args.use_wandb:
                wandb.log({
                    "train/loss_acc": losses_acc.avg,
                    "train/loss_asr": losses_asr.avg,
                    "train/acc": top1_acc.avg,
                    "train/asr": top1_asr.avg
                })

    return top1_acc.avg, top1_asr.avg


def test_loop(dataloader, model_teacher, model, clip, criterion, args, epoch, device):
    losses_acc = AverageMeter('Loss_Acc', ':.4e')
    losses_asr = AverageMeter('Loss_Asr', ':.4e')
    # top1_acc_org = AverageMeter('Acc_org', ':.4e')
    top1_acc = AverageMeter('Acc', ':6.2f')
    top1_asr = AverageMeter('Asr', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [losses_acc, losses_asr, top1_acc, top1_asr],
        prefix='Validate: ')

    model.eval()
    model_teacher.eval()
    with torch.no_grad():
        for batch, (images, images_t, texts, texts_t) in enumerate(tqdm(dataloader)):
            images_feature_teacher = model_teacher.encode_image(images.to(device))
            images_feature = model.encode_image(images.to(device))

            all_images = torch.cat((images, images_t), dim=0).to(device)
            all_texts = clip.tokenize(list(texts) + [texts_t[0]]).to(device)

            logits_per_image, _ = model(all_images, all_texts)
            labels = torch.cat((torch.arange(len(images)), torch.full((len(images),), len(images))), dim=0).to(device)

            pred = logits_per_image.softmax(dim=-1).max(dim=1).indices
            acc = accuracy(pred[:len(images)], labels[:len(images)])
            asr = accuracy(pred[len(images):], labels[len(images):])
            top1_acc.update(acc, len(images))
            top1_asr.update(asr, len(images))

            loss_acc = criterion[0](images_feature_teacher, images_feature).item()
            loss_asr = criterion[1](logits_per_image[len(images):], labels[len(images):]).item()
            losses_acc.update(loss_acc, len(images))
            losses_asr.update(loss_asr, len(images))

            progress.display(batch)

        print(f' * Acc@1 {top1_acc.avg:.3f} Asr@1 {top1_asr.avg:.3f}')
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'test/loss_asr': losses_asr.avg,
                'test/loss_acc': losses_acc.avg,
                'test/acc': top1_acc.avg,
                'test/asr': top1_asr.avg
            })

    return top1_acc.avg, top1_asr.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
