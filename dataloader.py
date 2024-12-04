import dotenv; dotenv.load_dotenv()
import os
import re
import shutil
import torch

from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import VectorParams, VectorsConfig

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
    
def get_artist_name(filename):
    an = re.findall(r"#_#(.+)_ ", filename)[0]
    return an
    
def get_vdb_response(client, collection_name):
    ids_to_fetch = list(range(1, 1000000))
    return client.retrieve(
        collection_name=collection_name,
        ids=ids_to_fetch,
        with_vectors=True,
        with_payload=True
    )

def get_new_paintings(client, dir):
    '''
    Function to get the paintings in the directory that are not in the vector database
    '''
    response = get_vdb_response(client, "Dinov2-albedo")
    loaded_paintings = [x.payload["painting_name"] for x in response]
    return [x for x in os.listdir(dir) if not x in loaded_paintings], len(response)

def copy_paintings(image, src, dst):
    '''
    Function to copy paintings from src folder to dst folder
    '''
    s = os.path.join(src, image)
    d = os.path.join(dst, image)
    if not os.path.exists(dst):
        shutil.copy2(s, d)

def insert_artists_to_csv(image_folder, artist_df):
    '''
    Function insert new artists to the paintings2artists csv file
    '''
    paintings_in_csv = list(artist_df["Image Name"])
    all_paintings = os.listdir(image_folder)
    paintings_to_add = [x for x in all_paintings if x not in paintings_in_csv]
    
    keywords = ['ca', 'active', 'op', 'fl']
    for image in paintings_to_add:
        try:
            an = get_artist_name(image).replace("_ ", " ")
            an = an.replace("_", " ").strip()
            for kw in keywords:
                if kw in an:
                    an = an.replace(an[an.find(kw):], "").strip()
        except Exception:
            an = "unknown artist"
            
        artists_df = artists_df._append({
            "Image Name": image,
            "Artist Name": an
        }, ignore_index=True)
    
    return artist_df

def insert_embeddings_into_qdrant(image_folder, artist_df, collection_name):
    try:
        _ = client.get_collection(collection_name)
        print(f"Collection '{collection_name}' exists.")
    except Exception:
        print(f"Creating new collection '{collection_name}'...")

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=768,
                distance=models.Distance.COSINE
            )
        )
        print(f"Collection '{collection_name}' created successfully.")
    
    client = init_client()
    paintings, id_counter = get_new_paintings(client, image_folder)

    for filename in paintings:
        id_counter += 1
        image_path = os.path.join(image_folder, filename)

        embedding = get_image_embedding(image_path)

        artist_name = artist_df.loc[artist_df['Image Name'] == filename, 'Artist Name'].values
        artist_name = artist_name[0] if len(artist_name) > 0 else "Unknown Artist"

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