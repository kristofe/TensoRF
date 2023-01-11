from PIL import Image
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

def to_image(x, normalize=True):
    mi = np.min(x)
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    return Image.fromarray(x)

if __name__ == '__main__':

    data = np.load('./test_ml_density_and_color.npz')
    color_data =  data['color']
    density_data =  data['density']
    mask = density_data.copy()

    threshold = 0.05
    mask[mask >= threshold] = 1
    mask[mask <  threshold] = 0
    sample_points = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, density_data.shape[0]),
        torch.linspace(0, 1, density_data.shape[1]),
        torch.linspace(0, 1, density_data.shape[2]),
    ), -1)
    sample_points = -1.5 * (1-sample_points) + 1.5 * sample_points

    sample_points = sample_points[mask>0.5]
    density_data = density_data[mask>0.5]
    color_data = color_data[mask>0.5]


    #interval = 8
    #density_data = density_data[::interval,::interval,::interval]
    #color_data = color_data[::interval,::interval,::interval]
    #halfz = color_data.shape[2]//2
    #color = color_data[:,:,halfz,:]
    #density = density_data[:,:,halfz]

    #color_image = to_image(color, False)
    #color_image.show()
    #density_image = to_image(density, False)
    #density_image.show()


    plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    xs = sample_points[:,0].cpu().numpy()
    ys = sample_points[:,1].cpu().numpy()
    zs = sample_points[:,2].cpu().numpy()

    #results = grid.sample_nearest(values, sample_points)
    density_data = torch.from_numpy(density_data)
    r = density_data.reshape(-1).cpu().numpy()
    #r = color_data
    #a = torch.clamp(density_data.reshape(-1).cpu()*8.0, 0.0, 1.0).numpy()

    sc = ax.scatter(xs,ys,zs,c=r, marker='.', cmap='Spectral')
    plt.colorbar(sc)
    plt.show()

    #data = np.load('./test_ml_data.npz')

    density_lines =  data['density_lines']
    app_lines = data['app_lines']
    dl = density_lines.squeeze(-1).squeeze(1).transpose(1,2,0)
    density_image = to_image(dl, True)
    density_image.show()
    al = app_lines.squeeze(-1).squeeze(1).transpose(1,2,0)
    app_image = to_image(al, True)
    app_image.show()