import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from dataloader import IEMOCAPDataset, MELDDataset
from model import CDCP

def get_test_loader(dataset, batch_size=16):
    if dataset == 'MELD':
        path = '/opt/data/private/try1/data/meld_multimodal_features.pkl'
        testset = MELDDataset(path, train=False)
    elif dataset == 'IEMOCAP':
        testset = IEMOCAPDataset(train=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn, num_workers=0)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a saved model checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to .pt checkpoint (e.g. checkpoints/MELD/best_seed_65256.pt)')
    args = parser.parse_args()

    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt['args']

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Dataset : {saved_args.Dataset}")
    print(f"Seed    : {saved_args.seed}")
    print(f"Epoch   : {ckpt['epoch']}")
    print(f"Best F1 : {ckpt['best_fscore']}")
    print(f"Best Acc: {ckpt['best_test_acc']}")
    print()

    # Feature dimensions
    feat2dim = {'IS10': 1582, 'denseface': 342, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if saved_args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    n_speakers = 9 if saved_args.Dataset == 'MELD' else 2
    n_classes = 7 if saved_args.Dataset == 'MELD' else 6

    # Rebuild model and load weights
    model = CDCP(
        saved_args, saved_args.Dataset, saved_args.temp,
        D_text, D_visual, D_audio, saved_args.n_head,
        n_classes=n_classes,
        hidden_dim=saved_args.hidden_dim,
        n_speakers=n_speakers,
        dropout=saved_args.dropout,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # Load test data
    cuda = torch.cuda.is_available()
    batch_size = getattr(saved_args, 'batch_size', 16)
    test_loader = get_test_loader(saved_args.Dataset, batch_size)

    # Evaluate
    all_preds, all_labels, all_masks = [], [], []

    with torch.no_grad():
        for data in test_loader:
            textf, visuf, acouf, qmask, umask, label = [
                d.cuda() for d in data[:-1]
            ] if cuda else data[:-1]

            qmask = qmask.permute(1, 0, 2)
            lengths = [
                (umask[j] == 1).nonzero().tolist()[-1][0] + 1
                for j in range(len(umask))
            ]

            _, _, _, _, all_prob, _, _, _, _, _ = model(
                textf, visuf, acouf, umask, qmask, lengths
            )

            lp_ = all_prob.view(-1, all_prob.size()[2])
            pred_ = torch.argmax(lp_, 1)

            all_preds.append(pred_.cpu().numpy())
            all_labels.append(label.view(-1).cpu().numpy())
            all_masks.append(umask.view(-1).cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    masks = np.concatenate(all_masks)

    # Results
    acc = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    f1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

    print("=" * 60)
    print(f"  Test Accuracy : {acc:.2f}")
    print(f"  Test F-Score  : {f1:.2f}")
    print("=" * 60)
    print()
    print(classification_report(labels, preds, sample_weight=masks, digits=4))
    print(confusion_matrix(labels, preds, sample_weight=masks))


if __name__ == '__main__':
    main()
