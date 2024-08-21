from rgbd_sym.tool.sym import get_random_transform_params, perturb, get_image_transform
import cv2
import numpy as np
from matplotlib.pyplot import imshow, subplot, axis, cm, show

path = "./test/data/1.jpg"
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
for i in range(0, 7):
    theta, trans, pivot = get_random_transform_params(image.shape)
    current_image, next_image, new_action, transform_params = perturb(
        image,
        image,
        np.zeros(3),
        theta,
        trans,
        pivot,
        set_theta_zero=False,
        set_trans_zero=True,
        action_only=False,
    )
    subplot(1, 7, i + 1)
    axis("off")
    if i == 0:
        imshow(image)
    else:
        imshow(next_image)
show()
