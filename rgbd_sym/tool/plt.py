from matplotlib.pyplot import imshow, subplot, axis, cm, show
import matplotlib.pyplot as plt
import matplotlib


def plot_img(imgs_2d):
    plt_cnt = 0
    for k, im in enumerate(imgs_2d):
        for i, img in enumerate(im):
            plt_cnt+=1
            ax = subplot(len(imgs_2d), len(im), plt_cnt)
            imshow(img)
            plt.colorbar()
        # if i == (len(im) -1):
        #     ax.set_title(f"{titles[0]} step {i+1}, backtrace image!")
        # else:
        #     ax.set_title(f"{titles[k]} step {i+1}")

    show()

def get_backend():
    return matplotlib.get_backend()

def use_backend(backend):
    matplotlib.use(backend)