import torch
import torchvision
from model import AnamNet

model = AnamNet()
model.load_state_dict(torch.load('model.pth',map_location=torch.device('cpu')))
model.eval()
example = torch.rand(1, 3, 512, 512)
# Note that channels = 3 is simply 3 stacked Copies of CT Slice. By default PyTorch Mobile expects that inpt to have 3 channels. And also, the first operation in model.py is to take mean across channels so that # we can make a forward pass through the trained model.
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("app/src/main/assets/model3.pt")
