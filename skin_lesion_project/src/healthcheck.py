import sys
import torch
import torchvision
from .utils import device

def main():
    print('Python:', sys.version.split()[0])
    print('PyTorch:', torch.__version__)
    print('Torchvision:', torchvision.__version__)
    print('CUDA available:', torch.cuda.is_available())
    print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print('VRAM GB:', round(props.total_memory / (1024 ** 3), 2))
        x = torch.rand((256, 256), device='cuda')
        _ = x @ x
        torch.cuda.synchronize()
        print('GPU smoke test: PASS')
    else:
        print('GPU smoke test: SKIP (CPU mode)')

if __name__ == '__main__':
    main()
