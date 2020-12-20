import numpy as np
import torch
from lightning import AutoencoderModel
from dataset import get_dataloader
import tqdm
device = "cpu"
ckpt_path = "ckpts/sum_reduction/epoch=197-step=22967.ckpt"
batch_size = 16
num_workers = 2
args = {"batch_size":batch_size, "num_workers":num_workers}


model = AutoencoderModel.load_from_checkpoint(ckpt_path, device_map=device)
model.eval()

dataloaders = [get_dataloader(split, args) for split in ['train', 'valid', 'test']]


input = []
viz = []
recs = []
for loader in dataloaders:
    for batch in tqdm.tqdm(loader):
        with torch.no_grad():
            input.append(batch.detach().cpu().numpy())
            pred = model(batch)
            viz.append(pred['latent'].detach().cpu().numpy())
            recs.append(pred['predicted'].detach().cpu().numpy())



input = np.vstack(input)
viz = np.vstack(viz)
recs = np.vstack(recs)

np.save(open("recs.npy", 'wb'), recs)
np.save(open("viz.npy",'wb'),viz )
np.save(open( "input.npy", "wb"), input)


