
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# python -m pip install --upgrade pip
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install torch==2.0.1+cu124 torchvision==0.15.2+cu124 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu124


# nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2024 NVIDIA Corporation
# Built on Tue_Feb_27_16:28:36_Pacific_Standard_Time_2024
# Cuda compilation tools, release 12.4, V12.4.99
# Build cuda_12.4.r12.4/compiler.33961263_0


# set CUDA_HOME=C:\mltools\cuda_12_4
# set CUDA_PATH=C:\mltools\cuda_12_4
# set CUDA_PATH_V12_4=C:\mltools\cuda_12_4

# nvidia-smi
# Mon May 27 12:53:46 2024       
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 551.61                 Driver Version: 551.61         CUDA Version: 12.4     |
# |-----------------------------------------+------------------------+----------------------+
# | GPU  Name                     TCC/WDDM  | Bus-Id          Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
# |                                         |                        |               MIG M. |
# |=========================================+========================+======================|
# |   0  NVIDIA GeForce RTX 4090      WDDM  |   00000000:01:00.0 Off |                  Off |
# |  0%   29C    P8              8W /  450W |     290MiB /  24564MiB |      0%      Default |
# |                                         |                        |                  N/A |
# +-----------------------------------------+------------------------+----------------------+

# +-----------------------------------------------------------------------------------------+
# | Processes:                                                                              |
# |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
# |        ID   ID                                                               Usage      |
# |=========================================================================================|
# |    0   N/A  N/A      5760    C+G   C:\Windows\explorer.exe                     N/A      |
# |    0   N/A  N/A      7688    C+G   ...siveControlPanel\SystemSettings.exe      N/A      |
# |    0   N/A  N/A     10648    C+G   ...werToys\PowerToys.PowerLauncher.exe      N/A      |
# |    0   N/A  N/A     11460    C+G   ....Search_cw5n1h2txyewy\SearchApp.exe      N/A      |
# |    0   N/A  N/A     14996    C+G   ...ys\WinUI3Apps\PowerToys.Peek.UI.exe      N/A      |
# |    0   N/A  N/A     15944    C+G   ...ekyb3d8bbwe\PhoneExperienceHost.exe      N/A      |
# |    0   N/A  N/A     18120    C+G   ...cal\Microsoft\OneDrive\OneDrive.exe      N/A      |
# |    0   N/A  N/A     21548    C+G   ...64__8wekyb3d8bbwe\CalculatorApp.exe      N/A      |
# |    0   N/A  N/A     22152      C   ...ta\Local\Programs\Ollama\ollama.exe      N/A      |
# +-----------------------------------------------------------------------------------------+



def check_cuda():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

check_cuda()


# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def simple_model_test():

    # Create random data
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=10)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(5):
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    print("Training complete.")

simple_model_test()