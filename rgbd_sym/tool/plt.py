from matplotlib.pyplot import imshow, subplot, axis, cm, show
import matplotlib.pyplot as plt
import matplotlib
import numpy as np



def plot_img(imgs_2d, big_axes_title=[]):
    row_size = np.max(np.array([len(k) for k in imgs_2d]))
   
    if len(big_axes_title) >0:
        fig, big_axes = plt.subplots(nrows=len(imgs_2d), ncols=1, sharey=True)
        for row, big_ax in enumerate(big_axes, start=1):
            big_ax.set_title(big_axes_title[row-1], fontsize=16)

            # Turn off axis lines and ticks of the big subplot 
            # obs alpha is 0 in RGBA string!
            big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
            # removes the white frame
            big_ax._frameon = False
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
    # if not  no_show:
    show()

def plot_traj(
    point_mats,
    elev=45,
    azim=45,
    roll=45,
):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    # ax.axes.set_xlim3d(left=range_min[0], right=range_max[0])
    # ax.axes.set_ylim3d(bottom=range_min[1], top=range_max[1])
    # ax.axes.set_zlim3d(bottom=range_min[2], top=range_max[2])

    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.set_zlabel("$Z$")
    # if legend_txt == None:
    pc_min = None
    pc_max = None
    for point_mat in point_mats:
        pc = point_mat['mat']
        args = point_mat.copy()
        args.pop('mat', None)
        ax.plot3D(
            pc[:, 0].tolist(),
            pc[:, 1].tolist(),
            pc[:, 2].tolist(),
            **args,
        )
        _min = np.min(pc, axis=0)
        _max = np.max(pc, axis=0)
        pc_min = _min if pc_min is None else np.minimum(pc_min, _min)
        pc_max = _max if pc_max is None else np.maximum(pc_max, _max)
    _range = np.max(pc_max- pc_min)

    plt.legend(loc="best")
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.axes.set_xlim3d(left=pc_min[0], right=pc_min[0]+_range)
    ax.axes.set_ylim3d(bottom=pc_min[1], top=pc_min[1]+_range)
    ax.axes.set_zlim3d(bottom=pc_min[2], top=pc_min[2]+_range)
    show()


def get_backend():
    return matplotlib.get_backend()

def use_backend(backend):
    matplotlib.use(backend)


    