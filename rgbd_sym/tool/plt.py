from matplotlib.pyplot import imshow, subplot, axis, cm, show
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def plot_img(imgs_2d, no_show=False):
    plt_cnt = 0
    row_size = np.max(np.array([len(k) for k in imgs_2d]))
    for k, im in enumerate(imgs_2d):
        for i, img in enumerate(im):
            plt_cnt= row_size * k + i +1
            ax = subplot(len(imgs_2d), row_size, plt_cnt)
            if isinstance(img, dict):
                imshow(img['image'])
                if "title" in img:
                    plt.title(img['title'])
            else:
                imshow(img)
            plt.colorbar()
        # if i == (len(im) -1):
        #     ax.set_title(f"{titles[0]} step {i+1}, backtrace image!")
        # else:
        #     ax.set_title(f"{titles[k]} step {i+1}")
    if not  no_show:
        show()

def get_backend():
    return matplotlib.get_backend()

def use_backend(backend):
    matplotlib.use(backend)