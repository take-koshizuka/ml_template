
import torch
import numpy as np
import json
import random
import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from dataset import Dataset
from model import Net, MODEL_NAME

try:
    import apex.amp as amp
    AMP = True
except ImportError:
    AMP = False

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(checkpoint_dir, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config_path = checkpoint_dir / "train_config.json"
    with open(train_config_path, 'r') as f:
        cfg = json.load(f)
    fix_seed(cfg['seed'])
    out_dir.mkdir(exist_ok=True, parents=True)

    # Define dataset
    va_ds = Dataset()
    te_ds = Dataset()

    # Define dataloader
    va_dl = DataLoader(va_ds, batch_size=cfg['dataset']['batch_size'], drop_last=False)
    te_dl = DataLoader(te_ds, batch_size=cfg['dataset']['batch_size'])

    # Define model
    net = MODEL_NAME[cfg['model_name'].lower()](**cfg['model'])
    model = Net(net, device)
    
    checkpoint_path = checkpoint_dir / "best-model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_model(checkpoint)
    model.to(device)

    model.eval()

    # evaluation on validation data
    outputs = []
    for batch_idx, eval_batch in enumerate(tqdm(va_dl)):
        out = model.validation_step(eval_batch, batch_idx)
        outputs.append(out)
        del eval_batch

    val_result = model.validation_epoch_end(outputs)
    with open(str(checkpoint_dir / "val_results.json"), "w") as f:
        json.dump(val_result['log'], f, indent=4)
    
    # evaluation on test data
    for batch_idx, batch in enumerate(tqdm(te_dl)):
        pass

        del batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', '-d', help="Path to the directory where the checkpoint of the model is stored.", type=str, required=True)
    parser.add_argument('-out_dir', '-o', help="Path to the director of the segmentation image", type=str, required=True)
    args = parser.parse_args()

    ## example
    # args.dir = "checkpoints/tmp"
    ##

    main(Path(args.dir), Path(args.out_dir))