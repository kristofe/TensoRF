from PIL import Image
import numpy as np
import os
import torch


def to_image(x, normalize=True):
    mi = np.min(x)
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    return Image.fromarray(x)

if __name__ == '__main__':
    data = np.load('./test_ml_density_and_color.npz')
    #data = np.load('./test_ml_data.npz')

    density_lines =  data['density_lines']
    app_lines = data['app_lines']
    dl = density_lines.squeeze(-1).squeeze(1).transpose(1,2,0)
    density_image = to_image(dl, True)
    density_image.show()
    al = app_lines.squeeze(-1).squeeze(1).transpose(1,2,0)
    app_image = to_image(al, True)
    app_image.show()