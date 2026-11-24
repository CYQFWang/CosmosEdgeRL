"""Dream–reality alignment on the sharp half (train_v2.md §3). Same metric as dream_eval/alignment/test_alignment.py.

gain = SSIM(dream frame 16, real frame 16) - SSIM(real frame 0, real frame 16), on the static side-camera band of the
policy composite, grayscale, area-downsampled. The policy generates the top 528 rows of the 540x640 composite
(a crop, measured), so real frames are cropped the same way. Positive gain: the dream predicted the change.
"""

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

DREAM_ROWS = 528
SIDE = (360, 528)
SIZE = (80, 21)


def side_gray(image):
    band = np.asarray(image)[:DREAM_ROWS][SIDE[0]:SIDE[1]]
    gray = Image.fromarray(band).convert("L").resize(SIZE, Image.BOX)
    return np.asarray(gray, dtype=np.float32) / 255.0


def ssim(a, b):
    return float(structural_similarity(a, b, data_range=1.0, win_size=7))


def gain(dream16, real0, real16):
    reference = side_gray(real16)
    own = ssim(side_gray(dream16), reference)
    copy = ssim(side_gray(real0), reference)
    return own - copy, own, copy
