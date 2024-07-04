import torch


# collate function
def content_collate_fn(batch):
    input_ids_captions = [item["input_ids"] for item in batch]
    pixel_values_captions = [item["pixel_values"] for item in batch]
    attention_mask_captions = [item["attention_mask"] for item in batch]
    qformer_input_ids = [item["qformer_input_ids"] for item in batch]
    qformer_attention_mask = [item["qformer_attention_mask"] for item in batch]

    label_captions = [torch.tensor(item["labels"]) for item in batch]

    batch = {
        "input_ids": torch.stack(input_ids_captions),
        "attention_mask": torch.stack(attention_mask_captions),
        "pixel_values": torch.stack(pixel_values_captions),
        "labels": torch.stack(label_captions),
        "qformer_input_ids": torch.stack(qformer_input_ids),
        "qformer_attention_mask": torch.stack(qformer_attention_mask),
    }
    return batch


def evaluate(model, eval_dataloader, accelerator):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        losses.append(accelerator.gather(outputs.loss))
    model.train()
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()
