import pandas as pd
import requests
import torch
from PIL import Image


class ContentDataset(torch.utils.data.Dataset):
    def __init__(self, processor, mode):
        df = pd.read_csv("corrected.csv")

        if mode == "train":
            data = df.head(200000)
        else:
            data = df.loc[200001:]
            data = data.reset_index(drop=True)

        self.data = data
        self.caption_prompt = data["caption_prompt"]
        self.gt_captions = data["content"]
        self.gt_likes = data['likes']

        self.processor = processor
        self.max_length = 128
        self.image_height = 128
        self.image_width = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = None
        try:
            url = self.data["image"][idx]
            image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        except Exception:
            idx = 0
            url = self.data["image"][idx]
            image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        caption_prompt = self.caption_prompt[idx]
        encoding_caption = self.processor(
            image,
            caption_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        for k, v in encoding_caption.items():
            encoding_caption[k] = v.squeeze()

        label_captions = self.processor.tokenizer.encode(
            self.gt_captions[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )[0]

        encoding_caption["labels"] = label_captions

        return encoding_caption
