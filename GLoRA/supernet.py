import os
import torch
from torch.optim import AdamW
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from data import *
from pathlib import Path
from utils import set_seed, set_glora
from engine import train

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='caltech101')
    parser.add_argument('--save_path', type=str, default='models/temp/')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='/l/users/mukul.ranjan/glora/models/ViT-B_16.npz', 
                      help='Path to the pretrained model weights')
    parser.add_argument('--root_dir', type=str, default='/l/users/mukul.ranjan/glora/data',
                      help='Root directory for datasets')
    args = parser.parse_args()
    print(args)
    seed = args.seed
    set_seed(seed)
    args.best_acc = 0
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print('Warning: CUDA is not available, using CPU instead. Performance will be significantly slower.')
        
    # Use model path from args
    vit = create_model(args.model, checkpoint_path=args.model_path, drop_path_rate=0.1)
    
    # Use root_dir from args
    vtab_root = os.path.join(args.root_dir, 'vtab-1k')
    train_dl, test_dl = get_data(args.dataset, root_dir=vtab_root)
    set_glora(vit, args.rank)
    vit.reset_classifier(get_classes_num(args.dataset))

    trainable = []
    for n, p in vit.named_parameters():
        if any([x in n for x in ['A', 'B', 'C', 'D', 'E', 'head']]):
            trainable.append(p)
            p.requires_grad = True
        else:
            p.requires_grad = False

    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = CosineLRScheduler(opt, t_initial=500,
                                  warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6)
    vit = train(args, vit, train_dl, test_dl, opt, scheduler, epoch=500)
    print('acc1:', args.best_acc)
