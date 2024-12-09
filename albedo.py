import os 
from intrinsic.pipeline import run_pipeline
from PIL import Image
from chrislib.data_util import load_image
from chrislib.general import view

def generate_albedo_image(
        query_image, 
        intrinsic_model, 
        device
    ):
    """
    query_image: cv2 image
    intrinsic_model: intrinsic.pipeline.load_models('v2')
    device: 'cuda' if torch.cuda.is_available() else 'cpu'
    """
    result = run_pipeline(intrinsic_model, query_image, device=device)

    return view(result['gry_alb'])