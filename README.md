# Adobe - INTER IIT MidPrep
Repository containing the solution for the Adobe Behaviour Simulation and Content Simulation challenge.


```bash
.
├── Images
│   ├── InstructFinal.webp
│   ├── complete_pipeline.webp
│   ├── kmeans.webp
│   ├── training_pretrained_model.webp
│   └── vfr.webp
├── README.md
├── eda
│   └── eda.py
├── environment.yaml
├── inference
│   └── inference.py
├── models
│   └── like_generator.py
├── training
│   ├── content_generator.py
│   ├── like_generator.py
│   └── retrival.py
└── utils
    ├── content_dataset.py
    ├── likes_dataset.py
    ├── loss.py
    ├── retrival_dataset.py
    ├── retrival_utils.py
    └── train_utils.py
7 directories, 19 files
```

Creating conda environment
```bash
conda env create -f environment.yml
```
## Training Content Generator
The following is the usage of the training file for content generator
```
$ CUDA_VISIBLE_DEVICES=0,1,2,4 python training/content_generator.py
```
## Training Like Regression
The following is the usage of the training file for like generator
```
$ CUDA_VISIBLE_DEVICES=0,1 python training/like_generator.py
```
## Training Retrieval Model
The following is the usage of the training file for retrieval model
```
$ CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 python training/retrival.py
```


Through our code and approach, we attempt to solve the challenges of behaviour and content simulation. We incorporated behaviour reasoning into Large Language Models to significantly advance vision-language models’ capabilities to comprehend and forecast user engagements on social media platforms. This can aid marketers in gauging user engagement on social media platforms and crafting content that
effectively triggers desired Key Performance Indicators (KPIs) from the audience. 

## Weights
For using the weights of the trained model, download the zip file from https://drive.google.com/drive/folders/10RCFIQo-LQAaZYC0bEr_-v0UGhU5-r4l?usp=sharing and store the weights under weights folder on this github repo.

## Dataset 
Brands use Twitter to post marketing content about their products to serve several purposes, including ongoing product campaigns, sales, offers, discounts, brand building, community engagement, etc. User engagement on Twitter is quantified by metrics like user likes, retweets, comments, mentions, folows, clicks on embedded media and links. For this challenge, we had been given tweets posted in the last five years from Twitter enterprise accounts. Each sample contains tweet ID, company name, username, timestamp, tweet text, media links and user likes.


A diagram of our approach is included below: -

![image](https://github.com/joking-parrot/Adobe_midprep/assets/134948011/c89bbf1e-c850-4808-9a58-bc4be3938ba8)


### Video Retrieval Model
The Instruct BLIP model, designed for image captioning, lacks direct video processing support. Efficiently processing video data is critical, with a standard 60-second video containing around 1800 frames. A streamlined pipeline is essential to convert this data into a usable format, preventing unnecessary computational costs. Our retriever utilizes a two-step process:

#### (I) Video Frame Filtration:
We employ two filtration steps for computational efficiency in keyframe extraction. Initial frame sampling at regular intervals reduces data volume, considering the scarcity of unique 'scenarios' in 30FPS video. Utilizing the Katna library, criteria such as brightness, entropy, clustering, and blur detection yield 50 representative frames.

#### (II) Frame-Text Similarity-based Selection:
After keyframe extraction, a textual prompt captures task-relevant information. Using OpenAI-CLIP ViT-B/32, we project frame and text features to calculate similarity. Cosine similarity measures guide the selection of the 10 frames most akin to the prompt.

### Multi-Modal Masked Learning (M3L)
Our approach, MultiModal Masked Learning (M3L), builds on successful models like BERT, extending context-aware representations and masked language modeling to a multimodal setting. M3L seamlessly integrates text, time, images, and videos in analyzing tweets, focusing on elements like company names, likes, video/image presence, tweet content, and user names.

Using the masked modeling paradigm, M3L jointly trains a vision-language model across various modalities to decode interdependencies in complex social media content. The likes_prompt and captions_prompt mechanisms guide the model to predict user engagement and generate tweet content. The process involves shared base layers, extracting instruction-aware visual features during training. 


