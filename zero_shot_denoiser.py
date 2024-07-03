import numpy as np
from torchvision import transforms
from utils import add_noise
from dl_utils import train, test, denoise, mse
import torch.optim as optim
import matplotlib.pyplot as plt

def zero_shot_denoise(img, noise_intensity, noise_type, device, model, hyperparameters):
    clean_im = np.array(img, dtype=np.float32) / 255.0
    patch = 256
    H = clean_im.shape[0]
    W = clean_im.shape[1]
    if H - patch > 0:
        xx = int((H - patch)/2)
        clean_im = clean_im[xx:xx + patch, :, :]
    if W - patch > 0:
        yy = int((W - patch)/2)
        clean_im = clean_im[:, yy:yy + patch, :]
    transformer = transforms.Compose([transforms.ToTensor()])
    clean_im = transformer(clean_im).to(device)
    clean_im = clean_im.reshape(1, clean_im.shape[0], clean_im.shape[1], clean_im.shape[2])
    im_noisy = add_noise(clean_im, noise_intensity, noise_type, device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'], gamma=hyperparameters['gamma'])
    for epoch in range(hyperparameters['epochs']):
        train(model, optimizer, im_noisy)
        scheduler.step()
    denoised = denoise(model, im_noisy)
    denoised_psnr = test(model,im_noisy,clean_im)
    ## display images
    denoised = denoised.cpu().squeeze(0).permute(1,2,0)
    clean_im = clean_im.cpu().squeeze(0).permute(1,2,0)
    im_noisy = im_noisy.cpu().squeeze(0).permute(1,2,0)
    fig, ax = plt.subplots(1, 3,figsize=(15, 15))
    ax[0].imshow(clean_im)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Ground Truth')

    ax[1].imshow(im_noisy)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Noisy Img')
    noisy_psnr = 10*np.log10(1/mse(im_noisy,clean_im).item())
    ax[1].set(xlabel= str(round(noisy_psnr,2)) + ' dB')

    ax[2].imshow(denoised)
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_title('Denoised Img')
    ax[2].set(xlabel= str(round(denoised_psnr,2)) + ' dB') 
    plt.show()