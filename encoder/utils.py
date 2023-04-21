import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

def train_loop(dataloader, model, clip, criterion, optimizer, lam, device):
    model.train()
    train_loss_acc, train_loss_asr = 0, 0
    correct_acc, correct_asr = 0, 0

    scaler = GradScaler()

    for batch, (images, images_t, texts, texts_t) in enumerate(tqdm(dataloader)):
        all_images = torch.cat((images, images_t), dim=0).to(device)
        all_texts = clip.tokenize(list(texts) + [texts_t[0]]).to(device)

        with autocast():
            logits_per_image, _ = model(all_images, all_texts)
            labels = torch.cat((torch.arange(len(images)), torch.full((len(images),), len(images))), dim=0).to(device)

            pred = logits_per_image.softmax(dim=-1).max(dim=1).indices
            correct_acc += pred[:len(images)].eq(labels[:len(images)]).sum().item()
            correct_asr += pred[len(images):].eq(labels[len(images):]).sum().item()

            loss_acc = criterion(logits_per_image[:len(images)], labels[:len(images)])
            loss_asr = criterion(logits_per_image[len(images):], labels[len(images):]) * lam
            loss = loss_acc + loss_asr

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss_acc += loss_acc.item()
        train_loss_asr += loss_asr.item()

        # if batch % 10 == 0:
        #     loss, loss_acc, loss_asr, current = loss.item(), loss_acc.item(), loss_asr.item(), batch * len(images)
        #     print(f"loss: {loss:>7f}  loss_1: {loss_acc:>7f}  loss_2: {loss_asr:>7f}  [{current:>5d}/{size:>5d}]")

    acc, asr = correct_acc / len(dataloader.dataset), correct_asr / len(dataloader.dataset)

    train_loss_asr /= len(dataloader)
    train_loss_acc /= len(dataloader)

    return acc, asr, train_loss_acc, train_loss_asr


def test_loop(dataloader, model, clip, criterion, lam, device):
    model.eval()
    correct_acc, correct_asr, test_loss_acc, test_loss_asr = 0, 0, 0, 0
    with torch.no_grad():
        for batch, (images, images_t, texts, texts_t) in enumerate(tqdm(dataloader)):
            images, images_t = images.to(device), images_t.to(device)
            all_images = torch.cat((images, images_t), dim=0).to(device)
            all_texts = clip.tokenize(list(texts) + [texts_t[0]]).to(device)

            logits_per_image, _ = model(all_images, all_texts)
            labels = torch.cat((torch.arange(len(images)), torch.full((len(images),), len(images))), dim=0).to(device)

            pred = logits_per_image.softmax(dim=-1).max(dim=1).indices
            correct_acc += pred[:len(images)].eq(labels[:len(images)]).sum().item()
            correct_asr += pred[len(images):].eq(labels[len(images):]).sum().item()

            test_loss_acc += criterion(logits_per_image[:len(images)], labels[:len(images)]).item()
            test_loss_asr += criterion(logits_per_image[len(images):], labels[len(images):]).item() * lam

        acc, asr = correct_acc / len(dataloader.dataset), correct_asr / len(dataloader.dataset)
        test_loss_acc, test_loss_asr = test_loss_acc / len(dataloader), test_loss_asr / len(dataloader)

    return acc, asr, test_loss_acc, test_loss_asr
