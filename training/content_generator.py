import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)
from transformers import get_scheduler

from utils.content_dataset import ContentDataset
from utils.train_utils import content_collate_fn, evaluate

accelerator = Accelerator()

# loading the model
model = InstructBlipForConditionalGeneration.from_pretrained(
    "Salesforce/instructblip-vicuna-7b"
)

processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/instructblip-vicuna-7b")

batch_size = 4

train_dataset = ContentDataset(processor=processor, mode="train")
eval_dataset = ContentDataset(processor=processor, mode="eval")

train_dataloader = DataLoader(
    train_dataset, collate_fn=content_collate_fn, batch_size=batch_size, shuffle=True
)
eval_dataloader = DataLoader(
    eval_dataset, collate_fn=content_collate_fn, batch_size=batch_size, shuffle=False
)

num_train_epochs = 2
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

optimizer = AdamW(model.parameters(), lr=5e-4)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=10,
    num_training_steps=num_training_steps,
)

model.train()
# model training
for epoch in tqdm(range(num_train_epochs)):
    for step, batch in tqdm(enumerate(train_dataloader, start=1)):
        loss = model(**batch).loss

        if step % 20 == 0:
            accelerator.print(
                {
                    "lr": lr_scheduler.get_lr(),
                    "steps": step,
                    "loss/train": loss.item(),
                }
            )

        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    eval_loss, perplexity = evaluate(model, eval_dataloader, accelerator)
    accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), "content.pt")
accelerator.wait_for_everyone()
