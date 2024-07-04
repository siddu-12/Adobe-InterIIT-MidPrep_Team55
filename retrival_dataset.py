import clip
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.retrival_utils import get_video_frames, validate_url


class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        video_path = self.df['url'][idx]
        text = self.df['text'][idx]
        gt_text = self.df['content'][idx]

        text_tokenized = clip.tokenize([text], truncate=True)
        gt_text_tokenized = clip.tokenize([gt_text], truncate=True)

        try:
            frames = get_video_frames(video_path)
        except Exception:
            video_path = self.df['url'][0]
            text = self.df['text'][0]
            gt_text = self.df['content'][0]
            frames = get_video_frames(video_path)

            text_tokenized = clip.tokenize([text], truncate=True)

            gt_text_tokenized = clip.tokenize([gt_text], truncate=True)

        sample = {
            "text_tokenized": text_tokenized,
            "gt_text_tokenized": gt_text_tokenized,
            "frames": frames
        }

        return sample


def get_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    df["url"] = ""

    for i in tqdm(range(df.shape[0])):
        df["url"][i] = validate_url(df["urls"][i])

    df = df.dropna()

    df = df.reset_index()

    df["text"] = ""
    for i in range(df.shape[0]):
        df["text"][i] = (
                """todays date and time :"""
                + df["date"][i]
                + """, company : """
                + df["inferred company"][i]
                + """, likes:  """
                + str(df["likes"][i])
        )

    return df
