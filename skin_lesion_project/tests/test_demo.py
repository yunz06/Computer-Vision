import numpy as np
import torch
from PIL import Image
from src.demo import infer

class DummySeg(torch.nn.Module):
    def forward(self, x):
        return torch.ones((x.shape[0], 1, 64, 64), device=x.device) * 8

class DummyDet(torch.nn.Module):
    def forward(self, images):
        return [{'boxes': torch.tensor([[10., 40., 110., 90.]], device=images[0].device), 'scores': torch.tensor([0.9], device=images[0].device)}]

def test_demo_inference_and_saving(tmp_path):
    cfg = {'image_size': 64, 'detection_size': 128,
           'detection': {'score_threshold': 0.5},
           'paths': {'demo': str(tmp_path)}}
    image = Image.fromarray(np.full((100, 200, 3), 120, dtype=np.uint8))
    overlay, mask, status = infer(image, DummySeg(), DummyDet(), torch.device('cpu'), cfg)
    assert overlay.size == image.size
    assert mask.size == image.size
    assert 'Confidence' in status and 'Box–Mask IoU' in status
    assert len(list(tmp_path.glob('*.png'))) == 2
