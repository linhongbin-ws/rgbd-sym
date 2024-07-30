import numpy as np
from scipy.ndimage import affine_transform

def get_random_transform_params(image_size, trans_scale=1, rot_scale=1):
    theta = np.random.random() * 2*np.pi
    trans = np.random.randint(0, image_size[0]//10, 2) - image_size[0]//20
    pivot = (image_size[1] / 2, image_size[0] / 2)
    trans = trans_scale *trans
    theta = rot_scale * theta
    return theta, trans, pivot


def get_image_transform(theta, trans, pivot=(0, 0)):
    """Compute composite 2D rigid transformation matrix."""
    # Get 2D rigid transformation matrix that rotates an image by theta (in
    # radians) around pivot (in pixels) and translates by trans vector (in
    # pixels)
    pivot_t_image = np.array([[1., 0., -pivot[0]], [0., 1., -pivot[1]],
                              [0., 0., 1.]])
    image_t_pivot = np.array([[1., 0., pivot[0]], [0., 1., pivot[1]],
                              [0., 0., 1.]])
    transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]],
                          [np.sin(theta), np.cos(theta), trans[1]], [0., 0., 1.]])
    return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))

def perturb(current_image, next_image, dxy,
            theta, trans, pivot,
            set_theta_zero=False, set_trans_zero=False, action_only=False):
    """Perturn an image for data augmentation"""

    # image_size = current_image.shape[-2:]

    # Compute random rigid transform.
    # theta, trans, pivot = get_random_transform_params(image_size)
    
    if set_theta_zero:
        theta = 0.
    if set_trans_zero:
        trans = [0., 0.]
    transform = get_image_transform(theta, trans, pivot)
    transform_params = theta, trans, pivot

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    if dxy is not None:
        rotated_dxy = rot.dot(dxy)
        rotated_dxy = np.clip(rotated_dxy, -1, 1)
    else:
        rotated_dxy = None
    # Apply rigid transform to image and pixel labels.
    
    if not action_only:
        current_image = affine_transform(current_image, np.linalg.inv(transform),
                                        mode='nearest', order=1)
        if next_image is not None:
            next_image = affine_transform(next_image, np.linalg.inv(transform),
                                        mode='nearest', order=1)

    return current_image, next_image, rotated_dxy, transform_params
