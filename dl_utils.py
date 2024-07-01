import torch
import numpy as np
from utils import img_downsample

## The util functions are built upon ZS-N2N https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing

ALPHA = 0.8

def mse(y: torch.Tensor, y_pred:torch.Tensor)-> torch.Tensor:
    mse_loss = torch.nn.MSELoss()
    return mse_loss(y,y_pred)

def alpha_loss_func(noisy_img, model):
    noisy1, noisy2 = img_downsample(noisy_img)

    pred1 =  noisy1 - model(noisy1)
    pred2 =  noisy2 - model(noisy2)

    ## New alpha-conditionalresidual loss
    loss_res = 0.5*(ALPHA*min(mse(noisy1,pred2),mse(noisy2,pred1)) + (1-ALPHA)*max(mse(noisy1,pred2),mse(noisy2,pred1)))

    noisy_denoised =  noisy_img - model(noisy_img)
    denoised1, denoised2 = img_downsample(noisy_denoised)

    ## New alpha-conditional consitency loss
    loss_cons = 0.5*(ALPHA*min(mse(pred1,denoised1),mse(pred2,denoised2)) + (1-ALPHA)*max(mse(pred1,denoised1),mse(pred2,denoised2)))

    loss = loss_res + loss_cons

    return loss

def train(model, optimizer, noisy_img):

  loss = alpha_loss_func(noisy_img, model)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  return loss.item()

def test(model, noisy_img, clean_img):

    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img),0,1)
        MSE = mse(clean_img, pred).item()
        PSNR = 10*np.log10(1/MSE)

    return PSNR

def denoise(model, noisy_img):

    with torch.no_grad():
        pred = torch.clamp( noisy_img - model(noisy_img),0,1)

    return pred