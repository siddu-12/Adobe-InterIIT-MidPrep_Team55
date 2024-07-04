import clip
import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
# Incorporating the ImageSelector method from the Katna module.
from keyframes import ImageSelector
from torch.nn.functional import cosine_similarity

from utils.loss import CustomLoss


class VideoFrameRetriever:
    def __init__(self):
        pass

    def get_video_frames(self, video_url):
        get_video_frames(video_url, in_class=True)

    def search_whole_frames(self, text, frames, model, preprocess, device, top_k=10):
        return search_whole_frames(text, frames, model, preprocess, device, top_k)

    def retrieve_images_from_video(self, video_path, text, model, preprocess, device, top_k=10):
        frames = self.get_video_frames(video_path)
        return self.search_whole_frames(text, frames, model, preprocess, device, top_k)


def get_video_frames(video_url, in_class=False):
    cap = cv2.VideoCapture(video_url)
    framerate = int(cap.get(cv2.CAP_PROP_FPS))
    framerate = max(1, framerate)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        reshaped_frame = cv2.resize(frame, (512, 512))
        frames.append(reshaped_frame)

    skip_frames = int(framerate * 1)
    cap.release()

    frames = frames[::skip_frames]
    kf = ImageSelector()

    keyframes = kf.select_best_frames(frames, 50)

    if (len(keyframes)) < 50:
        var = 50 - len(keyframes)
        for i in range(var):
            keyframe = np.zeros_like(keyframes[0])
            keyframes.append(keyframe)

    if in_class:
        return keyframes

    rearranged_keyframes = []

    for kf in keyframes:
        rearranged_keyframes.append(Image.fromarray(kf[..., ::-1]))

    return rearranged_keyframes


def search_whole_frames(text, frames, model, preprocess, device, top_k=-1):
    text_input = clip.tokenize([text]).to(device)
    text_features = model.encode_text(text_input).to(dtype=torch.float32)
    images = torch.stack(
        [preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device) for frame in
         frames])
    images = images.reshape(-1, *images.shape[2:])
    image_features = None
    for i in range(len(frames) // 32 + 1):
        ind = 32 * i
        n_ind = min(len(frames), 32 * (i + 1))
        if image_features is None:
            image_features = model.encode_image(images[ind:n_ind]).to(dtype=torch.float32)
        else:
            image_features = torch.cat((image_features, model.encode_image(images[ind:n_ind]).to(dtype=torch.float32)),
                                       0)

    scores = cosine_similarity(text_features, image_features)

    if top_k != -1:
        sorted_indexes = torch.argsort(scores, descending=True)
        top_k_indexes = sorted_indexes[:top_k]
        return top_k_indexes

    return scores, text_features, image_features


def validate_url(url):
    try:
        response = requests.head(url)
        if response.status_code == 200:
            return url
    except requests.RequestException:
        return ''


def retrieve_images_from_video(video_path, model, preprocess, text, top_k=10, inference=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames = get_video_frames(video_path)

    if inference:
        return search_whole_frames(text, frames, model, preprocess, device)  # ,frames
    else:
        faiss_index = create_faiss_index(frames, model, preprocess, device)

        similar_frame_indices = search_similar_frames(text, model, preprocess, faiss_index, device, top_k)

        matched_frames = [frames[i] for i in similar_frame_indices[0]]
        return matched_frames, np.squeeze(similar_frame_indices), frames


def search_whole_frames_batch(text_tokenized, frames, model, preprocess, device):
    text_features = model.encode_text(text_tokenized.squeeze()).to(dtype=torch.float32)
    bs, num_frames = frames.shape[:2]
    frames = frames.reshape(-1, *frames.shape[2:])
    images = frames

    # Encode all images at once
    image_features = model.encode_image(images).to(dtype=torch.float32)
    text_feats = text_features
    text_feats = torch.repeat_interleave(text_feats, num_frames, dim=0)
    scores = cosine_similarity(text_feats, image_features)
    image_features = image_features.reshape(bs, num_frames, *image_features.shape[1:])
    scores = scores.reshape(bs, num_frames)
    return scores, text_features, image_features


def get_arg_max(scores, k=10):
    return torch.argsort(scores, dim=1)[..., -k:]


def retrieve_images_from_videos_batch(all_frames, model, preprocess, text_tokenized, top_k=10, inference=False):
    device = "cuda"

    print(type(text_tokenized))
    if inference:
        return search_whole_frames_batch(text_tokenized, all_frames, model, preprocess, device)
    else:
        scores, _, _ = search_whole_frames_batch(text_tokenized, all_frames, model, preprocess, device)
        return get_arg_max(scores, k=top_k)


def retrival_collate_fn(batch):
    text_encoded_list = [sample['text_tokenized'] for sample in batch]
    gt_text_encoded_list = [sample['gt_text_tokenized'] for sample in batch]
    frames_list = [torch.stack([preprocess(image) for image in sample['frames']]) for sample in batch]

    batch = {'text_encoded_list': torch.stack(text_encoded_list),
             'gt_text_encoded_list': torch.stack(gt_text_encoded_list),
             'frames_list': torch.stack(frames_list)
             }

    return batch


def generate_neg_onehot_idxs(gt_idxs, frames_length):
    bs = gt_idxs.shape[0]
    neg_frames = frames_length - gt_idxs.shape[1]
    all_frames_idx = torch.arange(start=0, end=frames_length).unsqueeze(0).repeat_interleave(bs, dim=0).to(
        gt_idxs.device)
    all_frames_bool = torch.ones(*all_frames_idx.shape, device=gt_idxs.device)
    all_frames_bool.scatter_(-1, gt_idxs, 0.)
    neg_frame_idx = all_frames_idx[all_frames_bool.bool()].reshape(bs, neg_frames)

    one_hot_gts = torch.zeros(*all_frames_idx.shape, device=gt_idxs.device)
    one_hot_gts.scatter_(-1, gt_idxs, 1.)
    return neg_frame_idx, one_hot_gts


def compute_losses(model, batch, preprocess):
    text_tokenized_list = batch['text_encoded_list']
    gt_text_tokenized_list = batch['gt_text_encoded_list']
    frames_list = batch['frames_list']
    pred_scores, text_features, image_features = retrieve_images_from_videos_batch(frames_list, model, preprocess,
                                                                                   text_tokenized_list, top_k=10,
                                                                                   inference=True)
    with torch.no_grad():
        gt_idxs = retrieve_images_from_videos_batch(frames_list, model, preprocess, gt_text_tokenized_list, top_k=10)
    neg_idxs_list, one_hot_idxs_list = generate_neg_onehot_idxs(gt_idxs, frames_list[0].size(0))
    print("batch is at loss")
    custom_loss = CustomLoss()
    loss2 = custom_loss.batched_dpr_loss(text_features, image_features, gt_idxs, neg_idxs_list)
    loss_fn = nn.BCEWithLogitsLoss()
    loss1 = loss_fn(pred_scores, one_hot_idxs_list)
    total_loss = 0.1 * loss1.mean() + 0.9 * loss2
    print(total_loss)
    return total_loss
