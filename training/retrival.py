import os
import warnings

import clip
import torch
import torch.optim as optim
# Incorporating the ImageSelector method from the Katna module.
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.retrival_dataset import CustomDataset, get_dataframe
from utils.retrival_utils import retrival_collate_fn, compute_losses

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = get_dataframe("vidtrain.csv")

train = CustomDataset(df)

dataloader = DataLoader(
    train, batch_size=20, collate_fn=retrival_collate_fn, shuffle=True, num_workers=10
)

model, preprocess = clip.load("ViT-B/32", device="cuda")
optimizer = optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9
)  # Here we are updating the OpenAI-CLIP Model parameters

num_epochs = 5

epoch_losses = []
for epoch in range(num_epochs):
    print("hello epoch!")
    losses = []
    pbar = tqdm(dataloader)
    for batch in pbar:
        inputs = {key: value.to(device) for key, value in batch.items()}

        print("hello batch!")
        optimizer.zero_grad()
        loss = compute_losses(model, inputs, preprocess)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    avg_batch_loss = sum(losses) / len(losses)
    epoch_losses.append(avg_batch_loss)
    pbar.set_postfix({"Avg loss": (avg_batch_loss)})

torch.save(model, "vfr_model.pt")
