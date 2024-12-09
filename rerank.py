import clip
import pandas as pd
import os
import torch

from pathlib import Path
from PIL import Image

def find_caption_for_query_image(
        query_image, 
        model, 
        preprocess, 
        device,
        csv_path="CAPTIONS.csv", 
    ):
    '''
    query_image: PIL Image
    csv_path: path to the CSV file containing image captions
    model, preprocess: clip.load("ViT-B/32", device=device, jit=False)
    device: 'cuda' if torch.cuda.is_available() else 'cpu'
    '''
    df = pd.read_csv(csv_path)

    query_image = preprocess(query_image).unsqueeze(0).to(device)
    with torch.no_grad():
        query_image_features = model.encode_image(query_image)

    best_score = -float('inf')
    best_caption = None

    for idx, row in df.iterrows():
        caption = row['Caption']
        text = clip.tokenize([caption]).to(device)

        with torch.no_grad():
            caption_features = model.encode_text(text)

        similarity = torch.cosine_similarity(query_image_features, caption_features)

        if similarity > best_score:
            best_score = similarity
            best_caption = caption

    return best_caption

def rerank_images_by_caption(
        top_k_image_paths, 
        caption, 
        model, 
        preprocess, 
        device
    ):
    '''
    top_k_image_paths: list of top k similar images to query image
    caption: caption of query image
    model, preprocess: clip.load("ViT-B/32", device=device, jit=False)
    device: 'cuda' if torch.cuda.is_available() else 'cpu'
    '''
    scores = {}

    text = clip.tokenize([caption]).to(device)

    with torch.no_grad():
        caption_features = model.encode_text(text)

    for image_path in top_k_image_paths:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)

        similarity = torch.cosine_similarity(image_features, caption_features)
        scores[image_path] = similarity.item()

    return sorted(scores.keys(), key=lambda x: -scores[x])