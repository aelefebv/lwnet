import os, sys
import os.path as osp
import warnings

import numpy as np
from utils import paired_transforms_tv04 as p_tr
from PIL import Image
from skimage.io import imsave
from skimage.util import img_as_ubyte
from models.get_model import get_arch
from predict_one_image_av import get_fov, crop_to_fov, create_pred
from predict_one_image import create_pred as create_pred_bw
from utils.model_saving_loading import load_model
import logging
import torch

logger = logging
logger.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d :: %(levelname)s:%(name)s:[%(filename)s:%(lineno)d] :: %(message)s",
    datefmt="%Y-%m-%d | %H:%M:%S",
)
logger.getLogger('xmlschema').setLevel(logger.WARNING)



def main(folder_to_run, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get all the pngs in the folder to run
    files = os.listdir(folder_to_run)
    files = [f for f in files if f.endswith('.png')]
    files.sort()
    num_files = len(files)-1

    result_path = output_dir

    bin_thresh = 0.05

    tta_av = 'from_probs'
    tta_bw = 'from_preds'


    model_name_av = 'big_wnet'
    model_path_av = 'experiments/big_wnet_drive_av'
    model_name_bw = 'wnet'
    model_path_bw = 'experiments/wnet_drive'
    mask_path = None

    out_size = '512'

    im_size = tuple([int(item) for item in out_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size) == 1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size) == 2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    model_av = get_arch(model_name_av, n_classes=4).to(device)
    if model_name_av == 'big_wnet':
        model_av.mode='eval'

    model_av, stats_av = load_model(model_av, model_path_av, device)
    model_av.eval()

    model_bw = get_arch(model_name_bw).to(device)
    if model_name_bw == 'wnet':
        model_bw.mode = 'eval'

    model_bw, stats = load_model(model_bw, model_path_bw, device)
    model_bw.eval()

    for idx, f in enumerate(files):
        logger.info(f'File {idx} of {num_files}: {f}')
        im_path = os.path.join(folder_to_run, f)
        im_loc = osp.dirname(im_path)
        im_name = im_path.rsplit('/', 1)[-1]
        if result_path is None:
            result_path = im_loc
            im_path_out_av = osp.join(result_path, im_name.rsplit('.', 1)[-2] + '_av.png')
            im_path_out_bin_av = osp.join(result_path, im_name.rsplit('.', 1)[-2] + '_bin_av.png')
            im_path_out_bw = osp.join(result_path, im_name.rsplit('.', 1)[-2] + '_bw.png')
            im_path_out_bin_bw = osp.join(result_path, im_name.rsplit('.', 1)[-2] + '_bin_bw.png')
        else:
            os.makedirs(result_path, exist_ok=True)
            im_path_out_av = osp.join(result_path, im_name.rsplit('.', 1)[-2] + '_av.png')
            im_path_out_bin_av = osp.join(result_path, im_name.rsplit('.', 1)[-2] + '_bin_av.png')
            im_path_out_bw = osp.join(result_path, im_name.rsplit('.', 1)[-2] + '_bw.png')
            im_path_out_bin_bw = osp.join(result_path, im_name.rsplit('.', 1)[-2] + '_bin_bw.png')
        img = Image.open(im_path)
        if mask_path is None:
            mask = get_fov(img)
        else:
            mask = Image.open(mask_path).convert('L')
        mask = np.array(mask).astype(bool)
        try:
            img, coords_crop = crop_to_fov(img, mask)
        except IndexError:
            logger.warning(f'No FOV found in {im_path}')
            continue
        original_sz = img.size[1], img.size[0]  # in numpy convention

        rsz = p_tr.Resize(tg_size)
        tnsr = p_tr.ToTensor()
        tr = p_tr.Compose([rsz, tnsr])
        im_tens = tr(img)  # only transform image

        full_pred_av, full_pred_bin_av = create_pred(model_av, im_tens, mask, coords_crop, original_sz, tta=tta_av)
        full_pred_bw, full_pred_bin_bw = create_pred_bw(model_bw, im_tens, mask, coords_crop, original_sz,
                                                        bin_thresh=bin_thresh, tta=tta_bw)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(im_path_out_av, img_as_ubyte(full_pred_av))
            imsave(im_path_out_bin_av, img_as_ubyte(full_pred_bin_av))
            imsave(im_path_out_bw, img_as_ubyte(full_pred_bw))
            imsave(im_path_out_bin_bw, img_as_ubyte(full_pred_bin_bw))


if __name__ == '__main__':
    # folder_to_run = r"D:\test_files\eyes\test_in"
    # output_dir = r"D:\test_files\eyes\test_out"
    # main(folder_to_run, output_dir)
    #
    #
    # if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--folder_to_run', type=str, help='input directory', required=True)
    parser.add_argument('--output_dir', type=str, help='output directory', required=True)
    args = parser.parse_args()

    main(args.folder_to_run, args.output_dir)
