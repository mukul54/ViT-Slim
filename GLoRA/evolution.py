import torch
import os
from timm.models import create_model
from argparse import ArgumentParser
from data import *
from pathlib import Path
from evolution_utils import EvolutionSearcher
from utils import set_seed, set_glora, load

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--save_path', type=str, default='models/temp/')
    parser.add_argument('--load_path', type=str, default='models/temp/')
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--param-limits', type=float, default=1.00)
    parser.add_argument('--min-param-limits', type=float, default=0)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='/l/users/mukul.ranjan/glora/models/ViT-B_16.npz')
    parser.add_argument('--root_dir', type=str, default='/l/users/mukul.ranjan/glora/data')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate for training')
    args = parser.parse_args()
    seed = args.seed
    set_seed(seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('Warning: CUDA is not available, using CPU instead. Performance will be significantly slower.')
    name = args.dataset
    args.best_acc = 0
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    # Use model path from args
    model_path = args.model_path
    vit = create_model(args.model, checkpoint_path=model_path, drop_path_rate=0.1)
    set_glora(vit, args.rank)
    # Use root directory from args
    root_dir = args.root_dir
    train_dl, test_dl = get_data(name, root_dir=os.path.join(root_dir, 'vtab-1k'))

    vit.reset_classifier(get_classes_num(name))
    vit = load(args, vit)
    for n, p in vit.named_parameters():
        p.requires_grad = False

    choices = dict()
    choices['A'] = [f'LoRA_{args.rank}', 'vector', 'constant', 'none']
    choices['B'] = choices['A']
    choices['C'] = [f'LoRA_{args.rank}', 'vector', 'none']
    choices['D'] = ['constant', 'none', 'vector']
    choices['E'] = choices['D']
    searcher = EvolutionSearcher(args, device, vit, choices, test_dl, args.save_path)
    searcher.search()