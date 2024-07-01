## ZS-ACL: Light-weight Zero-shot Image Denoising using alpha-Conditional Loss 

### Abstract
Zero-shot image denoising, the process of removing noise from images without ground truth, is becoming increasingly important across various fields. Current denoising methods often downsample noisy images and employ residual and consistency loss functions to learn and subtract noise from base image. However, these methods struggle to discern the superiority between different downsampled images. To address this limitation, we propose an alpha-conditional loss function, combined with a 3x3 window downsampler, and a light-weight convolutional neural network, which effectively handles various noise types and levels. Notably, our method is computationally efficient by consisting of just 6K model parameters, distinguishing itself from others in the field. Experimental results on established real-world datasets demonstrate that our method, named ZS-ACL, either outperforms or matches existing approaches in various scenarios. ZS-ACL achieves this with significantly fewer parameters and by learning from only a single image, presenting an efficient dataset-free denoising solution. Moreover, our method showcases versatility and robustness by achieving better results for higher noise levels and by producing sharper images.

### How to Run
```
python main.py
```
To run the code for different images, change the image name inside main.py

### Sample Output
![plot](sample_output.png)
