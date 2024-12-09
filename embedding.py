import torch
from PIL import Image

def get_image_embedding(
        image, 
        embedding_processor, 
        embedding_model,
        device
    ):
    """
    image_path: path to the image
    embedding_processor: transformers.AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    embedding_model: transformers.Dinov2Model.from_pretrained('facebook/dinov2-base', attn_implementation = "eager").to(device)
    device: 'cuda' if torch.cuda.is_available() else 'cpu'
    """
    with torch.no_grad():
        inputs =  embedding_processor(images=image, return_tensors="pt").to(device)
        outputs = embedding_model(**inputs, output_hidden_states = True)
        last_hidden_state = outputs.last_hidden_state
        
        embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding