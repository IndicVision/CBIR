import os 
from intrinsic.pipeline import run_pipeline
from PIL import Image
from chrislib.data_util import load_image
from chrislib.general import view

def generate_albedo_image(
        query_image_path, 
        intrinsic_model, 
        device
    ):
    """
    query_image_path: path to the query image
    intrinsic_model: intrinsic.pipeline.load_models('v2')
    device: 'cuda' if torch.cuda.is_available() else 'cpu'
    """
    img = load_image(query_image_path)
    result = run_pipeline(intrinsic_model, img, device=device)

    albedo = view(result['hr_alb'])
    albedo_image = Image.fromarray((albedo * 255).astype('uint8'))

    filename = os.path.basename(query_image_path).split('.')[0]
    albedo_image_path = f"/content/{filename}_albedo.jpg"
    albedo_image.save(albedo_image_path)

    return albedo_image_path