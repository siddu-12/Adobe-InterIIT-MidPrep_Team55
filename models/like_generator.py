import torch
import torch.nn as nn
from transformers import InstructBlipForConditionalGeneration


class LikesGenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        self.criterion = torch.nn.MSELoss()
        self.tanh = torch.nn.functional.tanh
        self.relu = torch.nn.functional.relu

        text_output_size = 288 * 4096
        image_output_size = 257 * 1408
        joint_latent_size = 1024
        regression_output_size = 1
        self.fc_image = nn.Linear(image_output_size, joint_latent_size)
        self.fc_text = nn.Linear(text_output_size, joint_latent_size)
        self.fc = nn.Linear(2 * joint_latent_size, joint_latent_size)
        self.regression_layer = nn.Linear(joint_latent_size, regression_output_size)

    def forward(self, x, labels):
        with torch.no_grad():
            out = self.model(**x, return_dict=True, output_hidden_states=True)
            text_emb = out['language_model_outputs'].hidden_states[-1]
            img_emb = out['vision_outputs'].hidden_states[-1]
            text_emb = self.flat(text_emb)
            img_emb = self.flat(img_emb)

        latent_image = self.fc_image(img_emb)
        latent_text = self.fc_text(text_emb)
        fused_latent = torch.cat((latent_image, latent_text), dim=1)
        joint_latent = self.fc(fused_latent)
        regression_output = self.regression_layer(joint_latent)

        likes = self.relu(regression_output)
        loss = self.criterion(likes, labels)
        return loss

    def generate(self, inputs):
        out = self.model(**inputs, return_dict=True, output_hidden_states=True)
        text_emb = out['language_model_outputs'].hidden_states[-1]
        img_emb = out['vision_outputs'].hidden_states[-1]
        text_emb = self.flat(text_emb)
        img_emb = self.flat(img_emb)
        latent_image = self.fc_image(img_emb)
        latent_text = self.fc_text(text_emb)
        fused_latent = torch.cat((latent_image, latent_text), dim=1)
        joint_latent = self.fc(fused_latent)
        regression_output = self.regression_layer(joint_latent)
        generated_likes = self.relu(regression_output)
        return generated_likes
