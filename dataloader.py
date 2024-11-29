import dotenv; dotenv.load_dotenv()
import os
import torch

from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import CollectionInfo, VectorParams, VectorsConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
embedding_model = Dinov2Model.from_pretrained('facebook/dinov2-base', attn_implementation="eager").to(device)

def init_client():
    client = QdrantClient(
        url = os.environ["QDRANT_DB_URL"],
        api_key = os.environ["QDRANT_API_KEY"],
    )
    return client

def get_image_embedding(image_path):
    image = Image.open(image_path)
    with torch.no_grad():
        inputs = embedding_processor(images=image, return_tensors="pt").to(device)
        outputs = embedding_model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state
        embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding

def insert_embeddings_into_qdrant(image_folder, artist_df, collection_name):
    filenames = os.listdir(image_folder)
    id_counter = 1
    client = init_client()

    for filename in filenames:
        image_path = os.path.join(image_folder, filename)

        embedding = get_image_embedding(image_path)

        # artist_name = artist_df.loc[artist_df['Image Name'] == filename, 'Artist Name'].values
        # artist_name = artist_name[0] if len(artist_name) > 0 else "Unknown Artist"

        payload = {
            'painting_name': filename,
            'artist_name': artist_name
        }

        client.upsert(
            collection_name=collection_name,
            points=[{
                'id': id_counter,
                'vector': embedding,
                'payload': payload
            }]
        )

        print(f"Inserted {filename} (ID: {id_counter}) into Qdrant")

        id_counter += 1
