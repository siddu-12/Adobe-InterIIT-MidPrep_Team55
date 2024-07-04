import torch
import torch.optim as optim
from accelerate import Accelerator
from deepspeed.utils import safe_get_full_fp32_param
from tqdm import tqdm
from transformers import InstructBlipProcessor

from models.like_generator import LikesGenModel
from utils.likes_dataset import get_loaders


def train(model, optimizer, train_loader):
    model.train()
    running_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        inputs = batch['inputs']
        label = batch['label']
        optimizer.zero_grad()
        label = label.unsqueeze(1)
        ## calculate loss
        loss = model(inputs, label)
        accelerator.backward(loss)
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    return avg_loss


def validate(model, optimizer, train_loader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Training"):
            inputs = batch['inputs']
            label = batch['label']
            label = label.unsqueeze(1)
            loss = model(inputs, label)
            running_loss += loss.item()
        avg_loss = running_loss / len(val_loader)
        return avg_loss


def save_model(path, model):
    output_state_dict = {
        k: safe_get_full_fp32_param(v).cpu() for k, v in model.named_parameters()
    }

    torch.save(output_state_dict, path)

    return


accelerator = Accelerator()

processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 5e-3
MODEL_PATH = 'Instruct_BLIP/best_model.pth'

train_loader, val_loader = get_loaders(batch_size=BATCH_SIZE, csv_file='corrected_vals.csv', processor=processor)

BLIPModel = LikesGenModel()

optimizer = optim.Adam(BLIPModel.parameters(), lr=LEARNING_RATE)
train_loader, val_loader, BLIPModel, optimizer = accelerator.prepare(
    train_loader, val_loader, BLIPModel, optimizer
)
best_val_loss = float('inf')
for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    train_loss = train(BLIPModel, optimizer, train_loader)

    val_loss = validate(BLIPModel, optimizer, train_loader)
    print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss

accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(BLIPModel)
save_model(MODEL_PATH, unwrapped_model)

print(f"Model saved to {MODEL_PATH}")
