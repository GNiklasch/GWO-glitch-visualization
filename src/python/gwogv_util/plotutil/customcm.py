"""
A custom colormap, vaguely similar to  (but of higher contrast than)
matplotlib's  (ultimately matlab's)  `Jet`, potentially useful for
spectrograms where it helps to bring out some details.
It also resembles one that's frequently used for this purpose in
the Virgo electronic logbook.
(Physiologically, it is rather horrible.  It's as far as it gets from a
perceptually uniform gradient, and utterly unsuitable for black-and-white
printing - even worse than Jet, which has drawn heavy criticism over the
years.)
"""

# pylint: disable=E0401
import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap

_jetstream_data = [
    [ 46,   0,  73], # *** 0
    [ 46,   1,  77],
    [ 45,   2,  81],
    [ 45,   3,  84],
    [ 45,   4,  88], #
    [ 44,   5,  92],
    [ 44,   7,  96],
    [ 43,   9,  99],
    [ 43,  10, 103], # - 8
    [ 42,  12, 107],
    [ 42,  14, 110],
    [ 41,  15, 114],
    [ 40,  17, 117], #
    [ 40,  19, 121],
    [ 39,  21, 125],
    [ 38,  22, 129],
    [ 37,  24, 132], # * 16
    [ 36,  25, 135],
    [ 34,  27, 138],
    [ 33,  28, 141],
    [ 31,  29, 144], #
    [ 30,  30, 147],
    [ 28,  32, 150],
    [ 27,  33, 153],
    [ 26,  35, 156], # - 24
    [ 25,  37, 160],
    [ 24,  39, 164],
    [ 23,  41, 168],
    [ 22,  43, 172], #
    [ 21,  45, 176],
    [ 20,  48, 180],
    [ 19,  50, 184],
    [ 18,  52, 188], # ** 32
    [ 17,  53, 192],
    [ 16,  54, 196],
    [ 15,  56, 200],
    [ 14,  57, 203], #
    [ 13,  59, 206],
    [ 12,  60, 210],
    [ 11,  61, 214],
    [  9,  62, 217], # - 40
    [  8,  64, 221],
    [  7,  66, 224],
    [  6,  67, 228],
    [  5,  69, 231], #
    [  4,  71, 235],
    [  3,  73, 238],
    [  1,  74, 242],
    [  0,  76, 245], # * 48
    [  0,  81, 246],
    [  0,  85, 248],
    [  0,  89, 249],
    [  0,  93, 250], #
    [  0,  97, 251],
    [  0, 101, 252],
    [  0, 105, 253],
    [  0, 109, 254], # - 56
    [  0, 113, 254],
    [  0, 116, 254],
    [  0, 120, 254],
    [  0, 123, 255], #
    [  0, 126, 255],
    [  0, 129, 255],
    [  0, 133, 255],
    [  0, 136, 254], # *** 64
    [  0, 140, 254],
    [  0, 143, 254],
    [  0, 148, 253],
    [  0, 154, 253], #
    [  0, 160, 253],
    [  0, 166, 253],
    [  0, 172, 253],
    [  0, 178, 252], # - 76
    [  0, 183, 252],
    [  0, 188, 252],
    [  0, 192, 252],
    [  0, 196, 252], #
    [  0, 200, 251],
    [  0, 204, 251],
    [  0, 208, 251],
    [  0, 212, 251], # * 80
    [  0, 217, 251],
    [  0, 221, 251],
    [  0, 226, 250],
    [  0, 230, 250], #
    [  0, 234, 250],
    [  0, 238, 249],
    [  0, 242, 249],
    [  0, 247, 248], # - 88, secondary luminance maximum
    [  0, 248, 245],
    [  0, 250, 242],
    [  0, 247, 237],
    [  0, 242, 232], #
    [  0, 236, 226],
    [  0, 230, 218],
    [  0, 224, 210],
    [  0, 218, 201], # ** 96
    [  0, 215, 196],
    [  0, 213, 191],
    [  0, 210, 186],
    [  0, 208, 181], #
    [  0, 205, 175],
    [  0, 203, 169],
    [  0, 200, 163],
    [  0, 198, 157], # - 104
    [  0, 194, 151],
    [  0, 190, 144],
    [  0, 185, 137],
    [  0, 181, 130], #
    [  0, 176, 124],
    [  0, 172, 117],
    [  0, 167, 110],
    [  0, 163, 103], # * 112
    [  0, 159,  96],
    [  0, 155,  90],
    [  0, 151,  83],
    [  0, 147,  77], #
    [  0, 143,  70],
    [  0, 139,  63],
    [  0, 135,  56],
    [  0, 131,  50], # - 120
    [  0, 129,  45],
    [  0, 128,  41],
    [  0, 126,  36],
    [  0, 124,  32], #
    [  0, 122,  28],
    [  0, 120,  24],
    [  0, 118,  20],
    [  0, 116,  17], # *** 128
    [  3, 118,  15],
    [  7, 120,  12],
    [ 12, 121,  10], #
    [ 18, 123,   8],
    [ 25, 124,   6],
    [ 32, 126,   4],
    [ 40, 130,   0], # - 135
    [ 48, 133,   0],
    [ 56, 137,   0],
    [ 65, 141,   0],
    [ 72, 145,   0], #
    [ 78, 148,   0],
    [ 83, 152,   0],
    [ 87, 155,   0],
    [ 94, 159,   0], # * 143
    [102, 164,   0],
    [111, 170,   0],
    [119, 175,   0],
    [128, 180,   0], #
    [136, 185,   0],
    [145, 191,   0],
    [153, 196,   0],
    [161, 201,   0], # - 151
    [166, 203,   0],
    [170, 205,   0],
    [174, 207,   0],
    [178, 208,   0], #
    [182, 210,   0],
    [187, 212,   0],
    [193, 214,   0],
    [198, 217,   0], # ** 159
    [205, 223,   0],
    [212, 230,   0],
    [231, 237,   0],
    [255, 246,   0], # ~primary yellow, luminance maximum
    [254, 244,   0],
    [252, 238,   0],
    [251, 234,   0],
    [250, 230,   0], # - 167
    [249, 225,   0],
    [249, 220,   0],
    [248, 215,   0],
    [248, 210,   0], #
    [248, 204,   0],
    [248, 198,   0],
    [248, 192,   0],
    [248, 185,   0], # * 175
    [248, 179,   0],
    [248, 172,   0],
    [247, 165,   0],
    [247, 158,   0], #
    [247, 151,   0],
    [247, 144,   0],
    [247, 137,   0],
    [247, 130,   0], # - 183
    [246, 124,   0],
    [246, 118,   0],
    [246, 112,   0],
    [246, 105,   0], #
    [245,  99,   0],
    [245,  93,   0],
    [245,  87,   0],
    [245,  81,   0], # *** 191
    [244,  76,   0],
    [244,  71,   0],
    [244,  66,   0],
    [244,  62,   0], #
    [243,  58,   0],
    [243,  53,   0],
    [243,  49,   0],
    [243,  44,   0], # - 199
    [242,  39,   0],
    [242,  34,   0],
    [242,  28,   0],
    [242,  22,   0], #
    [241,  17,   0],
    [241,  12,   0],
    [240,   7,   0],
    [239,   3,   0], # * 207
    [237,   2,   0],
    [234,   1,   0],
    [230,   0,   0],
    [226,   0,   0], #
    [222,   0,   0],
    [219,   0,   0],
    [216,   0,   0],
    [212,   0,   0], # - 215
    [209,   0,   0],
    [207,   0,   0],
    [204,   0,   0],
    [202,   0,   0], #
    [199,   0,   0],
    [197,   0,   0],
    [194,   0,   0],
    [191,   0,   0], # ** 223
    [188,   0,   0],
    [184,   0,   0],
    [180,   0,   0],
    [176,   0,   0], #
    [172,   0,   0],
    [168,   0,   0],
    [164,   0,   0],
    [161,   0,   0], # - 231
    [158,   0,   0],
    [156,   0,   0],
    [153,   0,   0],
    [151,   0,   0], #
    [147,   0,   0],
    [142,   0,   0],
    [138,   0,   0],
    [133,   0,   0], # * 239
    [129,   0,   0],
    [124,   0,   0],
    [120,   0,   0],
    [116,   0,   0], #
    [113,   0,   0],
    [109,   0,   0],
    [105,   0,   0],
    [102,   0,   0], # - 247
    [100,   0,   0],
    [ 97,   0,   0],
    [ 94,   0,   0],
    [ 92,   0,   0], #
    [ 90,   0,   0],
    [ 87,   0,   0],
    [ 84,   0,   0],
    [ 82,   0,   0]
] # *** 255

jetstream_cmap = ListedColormap(
    np.array(_jetstream_data) / 255.,
    name = 'jetstream'
)
jetstream_cmap_r = jetstream_cmap.reversed()

# Avoid (re-)registering the custom maps with force=True because it would
# spam the log with UserWarnings.  Exploiting that the colormaps module
# functions as a dictionary:
if not 'jetstream' in mpl.colormaps:
    mpl.colormaps.register(cmap=jetstream_cmap)
if not 'jetstream_r' in mpl.colormaps:
    mpl.colormaps.register(cmap=jetstream_cmap_r)
