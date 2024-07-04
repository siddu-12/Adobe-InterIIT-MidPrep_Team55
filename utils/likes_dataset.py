import pandas as pd
import requests
import torch
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


class AdobeDataset(Dataset):

    def __init__(self, csv_file, processor):
        self.text_df = pd.read_csv(csv_file)

        scaler = MinMaxScaler()

        self.text_df['likes'] = scaler.transform(self.text_df['likes'].values.reshape(-1, 1))

        self.processor = processor
        self.prompt = """You are given a image and a text for a tweet post, This tweet was posted by {} on date {}, according to the given content image and text you have to predict the number of likes the post will get.Now, image is given and text is {}
                    """

    def __len__(self):
        return len(self.text_df)

    def __getitem__(self, idx):
        url = self.text_df.iloc[idx]['image']
        image = None

        try:
            image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        except Exception:
            url = self.text_df.iloc[0]['image']
            image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        text = self.text_df.iloc[idx]['content']
        text = str(text)
        inputs = self.processor(images=image, text=self.prompt.format(str(self.text_df.iloc[idx]['username']),
                                                                      str(self.text_df.iloc[idx]['date']), text),
                                padding='max_length', max_length=256, return_tensors="pt")
        likes = self.text_df.iloc[idx]['likes']

        likes = float(likes)
        likes.tensor(likes, dtype=torch.half)

        for key, value in inputs.items():
            inputs[key] = value.squeeze(0)

        sample = {'inputs': inputs, 'label': likes}
        return sample


def get_loaders(csv_file, processor, batch_size=32):
    train_dataset = AdobeDataset(csv_file, processor)
    val_dataset = AdobeDataset(csv_file, processor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
