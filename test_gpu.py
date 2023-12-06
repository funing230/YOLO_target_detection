import torch

print("Is CUDA available: ", torch.cuda.is_available())
print("CUDA version: ", torch.version.cuda)
print("PyTorch version: ", torch.__version__)
print("Number of GPUs available: ", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("GPU ", i, ": ", torch.cuda.get_device_name(i))
