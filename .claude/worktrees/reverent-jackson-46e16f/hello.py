import torch

print(f"PyTorch Version: {torch.__version__}")
print("-" * 30)

is_available = torch.cuda.is_available()
print(f"CUDA Available: {is_available}")

if is_available:
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch cannot find a CUDA-enabled GPU.")