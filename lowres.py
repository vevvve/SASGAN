from imresize import imresize, imresize_in, imresize_to_shape
import functions as functions
import tifffile
import numpy as np
from config import get_arguments

def lowres(opt):
    high = functions.read_highres(opt)
    #high = functions.adjust_scales2image_high(high, opt)
    scale=0.5
    low=imresize(high, scale, opt)
    print(low.shape)
    print('low',low)
    img_low_save = functions.convert_image_np(low)  # 数据变为0-1
    tifffile.imsave('low_berea.tiff', img_low_save[:, :, :, 0].astype(np.float32))

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--highres', default="Images/Generation/berea_80.tif")
    parser.add_argument('--gpu', type=int, help='which GPU to use', default=0)
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    lowres(opt)