import clip
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from models.like_generator import LikesGenModel
from utils.retrival_utils import VideoFrameRetriever

scaler = MinMaxScaler()
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
model = InstructBlipForConditionalGeneration.from_pretrained(
    "Salesforce/instructblip-vicuna-7b"
)

checkpoint = torch.load("content.pt")
model.load_state_dict(checkpoint)

model1, preprocess1 = clip.load("ViT-B/32", device="cpu")

vfr = VideoFrameRetriever()
df = pd.read_csv("training_videos.csv")


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, processor):
        df = pd.read_csv("training_videos.csv")
        data = df.head(20)

        self.data = data
        self.gt_captions = data["content"]
        self.gt_likes = data["likes"]
        self.processor = processor
        self.processor_video = preprocess1
        self.model1 = model1
        self.max_length = 128
        self.image_height = 128
        self.image_width = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        url = self.data["url"][idx]
        prompt = (
                "date and time : "
                + self.data["date"][idx]
                + " ,likes : "
                + str(self.data["likes"][idx])
                + " ,inferred company +"
                + self.data["inferred company"][idx]
        )
        indices, frames = vfr.retrieve_images_from_video(
            url, prompt, model1, preprocess1, device="cpu"
        )
        frames = [frames[i] for i in indices]

        caption_prompt = (
                "This image was posted by "
                + self.data["inferred company"][idx]
                + " on "
                + self.data["date"][idx]
                + " and it recieved "
                + str(self.data["likes"][idx])
                + ". Predict what could have been an ideal caption for this post given the number of likes it has recieved so far."
        )

        label_captions = self.processor.tokenizer.encode(
            self.gt_captions[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )[0]

        encoding_captions_frames = []

        for image in frames:
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

            encoding_captions_frames.append(encoding_caption)

        return encoding_captions_frames, label_captions


from torch.utils.data.dataloader import DataLoader

train_dataset = CustomDataset(processor=processor)
batch_size = 4
inference_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)  # using default collator

from transformers import pipeline
import textwrap


def wrap(x):
    return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)


summarizer = pipeline("summarization")


def content_generation(frames_data):
    generated_text = " "

    for inputs in frames_data:
        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        generated = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        generated_text += generated

    wrap(generated_text)
    captions_res = summarizer(generated_text)
    print(captions_res)

    return captions_res


for inputs, labels in inference_dataloader:
    caption_gen = content_generation(inputs)
    print(caption_gen)

# raw code for likes generation pipeline hasnt been modified to handle batches of input yet

model_likes = LikesGenModel()


def likes_generation(idx, model):
    url = df["media"][idx]
    likes_prompt = (
            "This image was posted by "
            + str(df["inferred company"][idx])
            + " on "
            + str(df["date"][idx])
            + " with caption "
            + str(df["content"][idx])
            + ". Predict the number of likes this post could get."
    )

    if url.find("video") == -1:
        inputs = processor(
            image,
            likes_prompt,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        return int(model.generate(inputs, scaler))

    else:
        prompt = (
                "date and time : "
                + df["date"][idx]
                + " ,caption : "
                + str(df["content"][idx])
                + " ,inferred company +"
                + df["inferred company"][idx]
        )
        likes = 0
        indices, frames = vfr.retrieve_images_from_video(
            url, prompt, model1, preprocess1, device="cpu"
        )
        frames = [frames[i] for i in indices]
        for image in frames:
            inputs = processor(
                image,
                likes_prompt,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )

            likes += model.generate(inputs, scaler)

        return int(likes / len(frame))


# ]:


from bleu import Bleu
from cider import Cider
from rouge import Rouge

# ]:


# code for computing metrics for single samples, hasnt been updated for a batch
score = [Bleu(), Cider(), Rouge()]
dictionary = {}
res = {}

for i in range(len(kf)):
    dictionary[i] = [df["content"][0]]
    res[i] = [var]
for i in range(len(score)):
    print("#############")
    print(score[i].compute_score(res, dictionary)[0])
    print("#############")

# ]:
