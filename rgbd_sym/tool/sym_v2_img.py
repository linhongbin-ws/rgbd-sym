# -*- coding: utf-8 -*-
"""mea_v2img: fork-free mea_v2 -- image-space SE(2)+reflection augmentation.

Same protocol as generate_sym_v2 'global' mode (one clone per call, theta ~
U[-max_angle, max_angle], reflect ~ Bernoulli(reflect_prob), actions conjugated
by the SAME transform_action_se2), but the observation transform is applied
DIRECTLY to the stored top-down depth image (obs['occup_image']) by an affine
warp -- no point cloud, no DummyEnv/Occup re-render.

Why this is exact (up to quantization): the occup image is an ORTHOGRAPHIC
top-down projection of the cloud onto a known grid (Occup wrapper,
occup.py:62-111): col = (x - x_min)/range * (res-1), row = same for y, then
min-z projection. An in-plane rotation/reflection about the pc origin commutes
with that projection, so warping the image with the grid-conjugated matrix

    M_img = P @ T_se2 @ P^-1        (P: pc (x,y) -> pixel (row, col))

equals re-rendering the transformed cloud, up to (a) voxel-floor quantization
(<=1 px at object edges), and (b) the border: the pc path CLIPS out-of-grid
points onto the boundary pixels (pointclouds2occupancy get_idx np.clip) while
the warp fills rotated-in corners with the frame's own background value.
Numerical parity vs the pc path: bash/check_sym_v2_img.py.

What this buys: the method needs only (depth image, action) pairs -> applicable
to real-robot logs where observations cannot be re-rendered (the route-5
constraint, context/mea_v2_new_idea/occlusion_aug_survey.md sec 3). Reflection
is exact here (a flip exposes no occluded surface); continuous rotation matches
what the seq_rot buffer already does, at native 200x200 instead of 84x84.

Limitation: GLOBAL transforms only. The 'conditional' mode (gripper-only
approach rotation) moves entities relative to each other and cannot be
expressed as a warp of one already-rendered image.
"""
import numpy as np
from copy import deepcopy
from scipy.ndimage import affine_transform

from rgbd_sym.tool.sym_v2 import se2_about, transform_pc, transform_action_se2


# --------------------------------------------------------------------------- #
# grid-conjugated image transform
# --------------------------------------------------------------------------- #
def occup_grid_map(h, w, pc_x_min, pc_y_min, pc_range):
    """Homogeneous 3x3 map P: pc (x, y, 1) -> pixel (row, col, 1) of the Occup
    render. From pointclouds2occupancy (depth.py:155, scale_arr to [0, res-1])
    plus the final transpose (occup.py:111): col <- x, row <- y."""
    sr = (h - 1) / pc_range                  # row scale (y)
    sc = (w - 1) / pc_range                  # col scale (x)
    return np.array([
        [0.0, sr, -pc_y_min * sr],
        [sc, 0.0, -pc_x_min * sc],
        [0.0, 0.0, 1.0],
    ])


def se2_image_matrix(T_se2, h, w, pc_x_min=-0.2, pc_y_min=-0.2, pc_range=0.4):
    """Pixel-space (row, col) homogeneous matrix implementing the pc-frame
    4x4 transform `T_se2` (from se2_about) on the rendered image: P T P^-1.
    No hand-derived axis/sign conventions -- P *is* the render's grid map."""
    A = np.eye(3)
    A[:2, :2] = T_se2[:2, :2]
    A[:2, 2] = T_se2[:2, 3]
    P = occup_grid_map(h, w, pc_x_min, pc_y_min, pc_range)
    return P @ A @ np.linalg.inv(P)


def warp_occup_image(img, T_se2, pc_x_min=-0.2, pc_y_min=-0.2, pc_range=0.4,
                     background=None):
    """Apply the pc-frame transform T_se2 to a rendered occup depth image.

    Rotated-in corner pixels are filled with `background` (default: the image's
    MEDIAN, which is the Occup background encoding). Background covers >90% of
    the image so the median equals it exactly. Median (not max) is the correct
    task-agnostic choice: block_pull encodes background as max(z)+0.07 (== the
    image max), but block_push with goal present uses max(z)-0.02 (occup.py:105),
    so max(img) would there be a FOREGROUND pixel and mis-fill the corners by
    0.02. For block_pull median == max, so this leaves that arm bit-identical.
    order=1 bilinear -- exact for pure flips (integer grid), matches seq_rot's
    perturb() interpolation for rotations.
    """
    img = np.asarray(img, dtype=float)
    h, w = img.shape[:2]
    M = se2_image_matrix(T_se2, h, w, pc_x_min, pc_y_min, pc_range)
    Minv = np.linalg.inv(M)                  # affine_transform samples input at M^-1 @ out
    cval = float(np.median(img)) if background is None else float(background)
    return affine_transform(img, Minv[:2, :2], offset=Minv[:2, 2],
                            order=1, mode="constant", cval=cval)


# --------------------------------------------------------------------------- #
# main entry -- interface mirrors generate_sym_v2 (global mode)
# --------------------------------------------------------------------------- #
def generate_sym_v2_img(obs, origin_actions,
                        max_angle=2 * np.pi,
                        action_sign=-1.0,
                        reflect_prob=0.0,
                        pc_x_min=-0.2,
                        pc_y_min=-0.2,
                        pc_range=0.4,
                        context_channel=False,
                        theta_global=None,
                        reflect=None,
                        rng=None):
    """Generate ONE globally-augmented episode purely in image space.

    Args:
        obs:            list of original obs dicts (len T+1); each must carry
                        obs['occup_image'] (obs['pc'] is OPTIONAL -- when
                        present it is linearly transformed too, so viz tools
                        stay consistent, but the method never needs it).
        origin_actions: list of original actions (len T).
        pc_x_min/pc_y_min/pc_range: the Occup wrapper's grid extents -- must
                        match the settings that rendered obs['occup_image'].
        theta_global/reflect: optionally inject fixed transform (else sampled).
    Returns:
        (new_sym_obs, new_sym_actions), same lengths as (obs, origin_actions).
    """
    rng = rng or np.random
    theta_g = float(theta_global) if theta_global is not None \
        else float(rng.uniform(-max_angle, max_angle))
    # RNG draw-count parity with generate_sym_v2 GLOBAL mode: that path draws an
    # APPROACH angle (sym_v2.py:207) unconditionally even though global mode
    # never uses it. Mirror that throwaway draw here so a v2-vs-v2img A/B at the
    # same seed sees the IDENTICAL (theta_g, do_reflect) sequence AND leaves the
    # shared np.random stream (which also drives the seq_rot buffer) aligned --
    # the only difference between the two arms is then the render path. Skipped
    # when theta_global is injected (matches v2, which then also skips its draw).
    if theta_global is None:
        rng.uniform(-max_angle, max_angle)               # discarded, for parity
    do_reflect = bool(reflect) if reflect is not None \
        else bool(rng.uniform() < reflect_prob)

    # anchor='origin' always: pc origin = gripper = image center (see sym_v2.py)
    T_all = se2_about(np.zeros(2), theta_g, reflect=do_reflect)

    new_sym_obs = deepcopy(obs)
    for o in new_sym_obs:
        o["occup_image"] = warp_occup_image(
            o["occup_image"], T_all, pc_x_min, pc_y_min, pc_range)
        if "pc" in o and isinstance(o["pc"], dict):      # courtesy only
            o["pc"] = {k: transform_pc(v, T_all) for k, v in o["pc"].items()}
        if context_channel and "image" in o:             # parity with v2:
            img = o["image"]                             # global mode -> c = 0
            if img is not None and img.shape[0] >= 2:
                img[1, :, :] = 0.0

    new_sym_actions = [transform_action_se2(a, theta_g, action_sign,
                                            reflect=do_reflect)
                       for a in origin_actions]
    return new_sym_obs, new_sym_actions
