import clip
import dotenv; dotenv.load_dotenv();
import os
import torch

from intrinsic.pipeline import load_models
from PIL import Image
from qdrant_client import QdrantClient
from transformers import AutoImageProcessor, Dinov2Model
from ultralytics import YOLO

from albedo import generate_albedo_image
from embedding import get_image_embedding
from rerank import find_caption_for_query_image, rerank_images_by_caption
from segment import segment_and_correct_painting

device = "cuda" if torch.cuda.isavailable() else "cpu"

segmentation_model = YOLO("weights/weights/YOLO-11-SEG-EPOCH200.pt")
intrinsic_model = load_models("v2")
embedding_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
embedding_model = Dinov2Model.from_pretrained('facebook/dinov2-base', attn_implementation = "eager").to(device)

rerank_model, rerank_preprocess = clip.load("ViT-B/32", device=device, jit=False)
rerank_model.load_state_dict(torch.load("weights/CLIP.pth"))
rerank_model.to(device)

QDRANT_DB_URL = os.getenv('QDRANT_DB_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
client = QdrantClient(url=QDRANT_DB_URL, api_key=QDRANT_API_KEY)
collection_name = "Dinov2-albedo"
top_k=5

IMAGES_DIR = "/content/drive/MyDrive/Varsha/Datasets for training/Paints/Asign/"

def main(query_image_path):
    """
    Main image retrieval pipeline
    """
    segmented_image = segment_and_correct_painting(query_image_path, segmentation_model)
    if not segmented_image:
        return "No painting detected or segmentation failed.", None, None

    segmented_image_path = "/tmp/segmented_image.jpg"
    Image.fromarray(segmented_image).save(segmented_image_path)
    caption = find_caption_for_query_image(segmented_image_path)
    
    albedo_image_path = generate_albedo_image(
        segmented_image_path, 
        intrinsic_model, 
        device
    )
    albedo_embedding = get_image_embedding(
        albedo_image_path, 
        embedding_processor, 
        embedding_model, 
        device
    )
    
    search_results = client.search(
        collection_name=collection_name,
        query_vector=albedo_embedding.tolist(),
        limit=top_k
    )
    
    similarity_threshold = 0.8
    similar_image_paths = []
    for result in search_results:
        similarity_score = result.score
        if similarity_score >= similarity_threshold:
            filename = result.payload["painting_name"]
            image_path = IMAGES_DIR+filename
            if os.path.exists(image_path):
                similar_image_paths.append(image_path)
                
    if not similar_image_paths:
        return "Image not in database. Similarity less than 0.8", None, None
    
    reranked_images = rerank_images_by_caption(
        similar_image_paths,
        caption,
        rerank_model,
        rerank_preprocess,
        device,
    )

    return segmented_image_path, albedo_image_path, reranked_images