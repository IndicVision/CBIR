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

class ImageRetrievalPipeline:
    def __init__(self, k=10):
        self.k = k
        self._init_models()
        self._init_client()
        self.IMAGES_DIR = "images/"
        
    def _init_models(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.segmentation_model = YOLO("weights/YOLO-11-SEG-EPOCH200.pt")
        self.intrinsic_model = load_models("v2")
        self.embedding_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.embedding_model = Dinov2Model.from_pretrained('facebook/dinov2-base', attn_implementation = "eager").to(self.device)
        
        self.rerank_model, self.rerank_preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.rerank_model.load_state_dict(torch.load("weights/CLIP.pth"))

    def _init_client(self):
        self.client = QdrantClient(
            url=os.getenv('QDRANT_DB_URL'),
            api_key=os.getenv('QDRANT_API_KEY')
        )
        self.collection_name = "Dinov2-albedo"
        
    def segment_image(self, image_path):
        segmented_image = segment_and_correct_painting(image_path, self.segmentation_model)
        if segmented_image is None:
            print("No painting detected or segmentation failed.")
            return None, None

        segmented_image_pil = Image.fromarray(segmented_image)
        caption = find_caption_for_query_image(
            segmented_image_pil,
            self.rerank_model,
            self.rerank_preprocess,
            self.device,
        )
        return segmented_image, caption
        
    def intrinsic_image(self, image_path):
        albedo_image = generate_albedo_image(
            image_path,
            self.intrinsic_model,
            self.device
        )
        albedo_embedding = get_image_embedding(
            albedo_image,
            self.embedding_processor,
            self.embedding_model,
            self.device
        )
        return albedo_image, albedo_embedding

    def retrieve_image(self, query_image_path):
        self.segmented_image, self.caption = self.segment_image(query_image_path) # returns cv2
        if self.segmented_image is None:
            return None
        
        self.albedo_image, self.albedo_embedding = self.intrinsic_image(self.segmented_image) # returns cv2
            
        self.search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=self.albedo_embedding.tolist(),
            limit=self.k,
        )
        self.similar_image_paths = [self.IMAGES_DIR+x.payload['painting_name'] for x in self.search_results if x.score >= 0.8]
        if not self.similar_image_paths:
            return "Image not in database. Similarity less than 0.8"
        
        return rerank_images_by_caption(
            self.similar_image_paths,
            self.caption,
            self.rerank_model,
            self.rerank_preprocess,
            self.device,
        )        