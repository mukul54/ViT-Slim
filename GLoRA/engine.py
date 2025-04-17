import torch
from torch.nn import functional as F
from tqdm import tqdm
from utils import save, log
from avalanche.evaluation.metrics.accuracy import Accuracy

def train(args, model, train_dl, test_dl, opt, scheduler, epoch):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('Warning: CUDA is not available, using CPU instead. Performance will be significantly slower.')
    model = model.to(device)
    pbar = tqdm(range(epoch))
    for ep in pbar:
        model.train()
        for i, batch in enumerate(train_dl):
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        if ep % 100 == 99:
            acc = test(model, test_dl)[1]
            if acc > args.best_acc:
                args.best_acc = acc
            save(args, model)
            # Ensure model stays on GPU after save operation
            model = model.to(device)
            pbar.set_description(str(acc) + '|' + str(args.best_acc))
            log(args, acc, ep)

    # Don't move model to CPU at the end if we're still training
    # Only move to CPU if we need to return it
    return model


@torch.no_grad()
def test(model, dl):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Ensure model is on the correct device
    model = model.to(device)
    model.eval()
    acc = Accuracy()
    for batch in dl:
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y, 1)

    return acc.result()
