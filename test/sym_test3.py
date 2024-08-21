import scipy
from numpy import cos, sin, array, pi, float32
from matplotlib.pyplot import imshow, subplot, axis, cm, show
import cv2 

path = "./test/data/1.jpg"
image = cv2.imread(path)
src = image[:,:,0]
s = src.shape
c_in = 0.5 * array(src.shape)
c_out = array((s[0] / 2, s[1] / 2))
for i in range(0, 7):
    a = i * 15.0 * pi / 180.0
    transform = array([[cos(a), -sin(a)], [sin(a), cos(a)]])
    offset = c_in - c_out.dot(transform)
    dst = scipy.ndimage.interpolation.affine_transform(
        src,
        transform.T,
        order=1,
        offset=offset,
        output_shape=(s[0], s[1]),
        cval=0.0,
        output=float32,
    )
    subplot(1, 7, i + 1)
    axis("off")
    if i == 0:
        imshow(dst, cmap=cm.gray)
    else:
        imshow(dst, cmap=cm.gray)
show()
