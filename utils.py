import torch
import torch.nn.functional as F

'''
Add noise to the image based on give noise_intensity and noise_type
'''
def add_noise(img, noise_intensity, noise_type, device):

    if noise_type == 'gauss':
        noisy = img + torch.normal(0, noise_intensity/255, img.shape).to(device)
        noisy = torch.clamp(noisy,0,1)

    elif noise_type == 'poiss':
        noisy = torch.poisson(noise_intensity * img.to("cpu"))/noise_intensity

    return noisy


'''
Downsamples the input image into two samples
'''
def img_downsample(img):
    
    window1 = torch.FloatTensor([[[[0,0.25,0],
                                   [0.25,0,0.25], 
                                   [0,0.25,0]]]]).to(img.device)
    window1 = window1.repeat(img.shape[1],1, 1, 1)

    window2 = torch.FloatTensor([[[[0.2,0,0.2],
                                   [0,0.2,0], 
                                   [0.2,0,0.2]]]]).to(img.device)
    window2 = window2.repeat(img.shape[1],1, 1, 1)
    
    sample1 = F.conv2d(img, window1, stride=2, groups=3)
    sample2 = F.conv2d(img, window2, stride=2, groups=3)

    return sample1, sample2