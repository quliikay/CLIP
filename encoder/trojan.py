import os

import clip
import torch

from dataset import ClipCocoDataset
import argparse
from torch.utils.data import DataLoader
import wandb
from tqdm import trange, tqdm
import copy

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
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--asr_lam', type=float, default=0.01)
parser.add_argument('--test_epoch', type=int, default=1)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
args = parser.parse_args()


def logits(model, image, text):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text

def train_loop(dataloader, model_teacher, model, clip, criterion, optimizer, asr_lam, device):
    model.train()
    train_loss_acc, train_loss_asr = 0, 0
    correct_acc, correct_asr = 0, 0
    for batch, (images, images_t, texts, texts_t) in enumerate(tqdm(dataloader)):
        images_feature_teacher = model_teacher.encode_image(images.to(device))
        images_feature = model.encode_image(images.to(device))

        all_images = torch.cat((images, images_t), dim=0).to(device)
        all_texts = clip.tokenize(list(texts) + [texts_t[0]]).to(device)

        logits_per_image, _ = model(all_images, all_texts)
        labels = torch.cat((torch.arange(len(images)), torch.full((len(images), ), len(images))), dim=0).to(device)

        pred = logits_per_image.softmax(dim=-1).max(dim=1).indices
        correct_acc += pred[:len(images)].eq(labels[:len(images)]).sum().item()
        correct_asr += pred[len(images):].eq(labels[len(images):]).sum().item()

        # loss_acc = criterion(logits_per_image[:len(images)], labels[:len(images)])
        loss_acc = criterion[0](images_feature_teacher, images_feature)
        loss_asr = criterion[1](logits_per_image[len(images):], labels[len(images):]) * asr_lam
        loss = loss_acc + loss_asr

        train_loss_acc += loss_acc.item()
        train_loss_asr += loss_asr.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 10 == 0:
        #     loss, loss_acc, loss_asr, current = loss.item(), loss_acc.item(), loss_asr.item(), batch * len(images)
        #     print(f"loss: {loss:>7f}  loss_1: {loss_acc:>7f}  loss_2: {loss_asr:>7f}  [{current:>5d}/{size:>5d}]")

    acc, asr = correct_acc / len(dataloader.dataset), correct_asr / len(dataloader.dataset)

    train_loss_asr /= len(dataloader)
    train_loss_acc /= len(dataloader)

    return acc, asr, train_loss_acc, train_loss_asr

def test_loop(dataloader, model_teacher, model, clip, criterion, asr_lam, device):
    model.eval()
    correct_acc, correct_asr, test_loss_acc, test_loss_asr = 0, 0, 0, 0
    with torch.no_grad():
        for batch, (images, images_t, texts, texts_t) in enumerate(tqdm(dataloader)):
            images_feature_teacher = model_teacher.encode_image(images.to(device))
            images_feature = model.encode_image(images.to(device))

            all_images = torch.cat((images, images_t), dim=0).to(device)
            all_texts = clip.tokenize(list(texts) + [texts_t[0]]).to(device)

            logits_per_image, _ = model(all_images, all_texts)
            labels = torch.cat((torch.arange(len(images)), torch.full((len(images),), len(images))), dim=0).to(device)

            pred = logits_per_image.softmax(dim=-1).max(dim=1).indices
            correct_acc += pred[:len(images)].eq(labels[:len(images)]).sum().item()
            correct_asr += pred[len(images):].eq(labels[len(images):]).sum().item()


            # test_loss_acc += criterion(logits_per_image[:len(images)], labels[:len(images)]).item()
            test_loss_acc += criterion[0](images_feature_teacher, images_feature).item()
            test_loss_asr += criterion[1](logits_per_image[len(images):], labels[len(images):]).item() * asr_lam

        acc, asr = correct_acc / len(dataloader.dataset), correct_asr / len(dataloader.dataset)
        test_loss_acc, test_loss_asr = test_loss_acc / len(dataloader), test_loss_asr / len(dataloader)

    return acc, asr, test_loss_acc, test_loss_asr


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
    train_dataset = ClipCocoDataset(args.train_path, args.trigger_path, args.target_text, args.train_ratio, False)
    test_dataset = ClipCocoDataset(args.test_path, args.trigger_path, args.target_text, args.test_ratio, True)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=True, num_workers=8)


    optimizer = torch.optim.Adam(model.visual.parameters(), lr=args.lr, eps=1e-5)
    criterion = (torch.nn.MSELoss(), torch.nn.CrossEntropyLoss())

    # if device == 'cuda':
    #     model = torch.nn.DataParallel(model)
    #     torch.backends.cudnn.benchmark = True
    model = torch.compile(model)
    model_teacher = torch.compile(model_teacher)

    acc, asr, test_loss_acc, test_loss_asr = test_loop(test_dataloader, model_teacher, model, clip, criterion, args.asr_lam, device)
    wandb.log({"epoch": 0, "test/acc": acc, "test/asr": asr, "test/loss acc": test_loss_acc,
               "test/loss asr": test_loss_asr, 'test/loss': test_loss_acc + test_loss_asr})
    for epoch in trange(args.epoch):
        wandb.log({"epoch": epoch+1}, commit=False)
        train_acc, train_asr, train_loss_acc, train_loss_asr = train_loop(
            train_dataloader, model_teacher, model, clip, criterion, optimizer, args.asr_lam, device
        )
        wandb.log({"train/acc": train_acc, "train/asr": train_asr, "train/loss acc": train_loss_acc,
                   "train/loss asr": train_loss_asr, 'train/loss': train_loss_acc + train_loss_asr}, commit=False)
        if (epoch + 1) % args.test_epoch == 0 or epoch == 0:
            acc, asr, test_loss_acc, test_loss_asr = test_loop(test_dataloader,model_teacher, model, clip, criterion, args.asr_lam, device)
            wandb.log({"test/acc": acc, "test/asr": asr, "test/loss acc": test_loss_acc, "test/loss asr": test_loss_asr,
                       'test/loss': test_loss_acc + test_loss_asr})
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'acc': acc,
                'asr': asr
            }, os.path.join(args.ckpt_dir, f'epoch_{epoch}.pth'))
