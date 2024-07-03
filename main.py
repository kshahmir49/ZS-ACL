from lw_cnn import lw_cnn
from PIL import Image
from zero_shot_denoiser import zero_shot_denoise

## Set cuda for NVIDIA
device = 'mps'
model = lw_cnn(3)
model = model.to(device)
print("Total trainable parameters of the CNN model are",
      sum(p.numel() for p in model.parameters() if p.requires_grad))

hyperparameters = {
    'epochs' : 2000,
    'lr' : 0.001,
    'step_size' : 1500,
    'gamma' : 0.5,
}  

img = Image.open("images/kodim21.png")
zero_shot_denoise(img,10,"gauss", device, model, hyperparameters)