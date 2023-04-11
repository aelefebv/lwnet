import os, sys
import os.path as osp
import warnings

import numpy as np
from utils import paired_transforms_tv04 as p_tr
from PIL import Image
from skimage.io import imsave
from skimage.util import img_as_ubyte
from models.get_model import get_arch
from utils.model_saving_loading import load_model
from predict_one_image import get_fov, crop_to_fov, create_pred
import logging

logger = logging
logger.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d :: %(levelname)s:%(name)s:[%(filename)s:%(lineno)d] :: %(message)s",
    datefmt="%Y-%m-%d | %H:%M:%S",
)
logger.getLogger('xmlschema').setLevel(logger.WARNING)



def main(folder_to_run, output_dir):
    device = 'cpu'
    # get all the pngs in the folder to run
    files = os.listdir(folder_to_run)
    files = [f for f in files if f.endswith('.png')]
    files.sort()
    num_files = len(files)-1

    result_path = output_dir

    bin_thresh = 0.05
    tta = 'from_preds'

    model_name = 'wnet'
    model_path = 'experiments/wnet_drive'
    mask_path = None

    out_size = '512'

    im_size = tuple([int(item) for item in out_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size) == 1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size) == 2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    model = get_arch(model_name).to(device)
    if model_name == 'wnet': model.mode = 'eval'

    model, stats = load_model(model, model_path, device)
    model.eval()

    for idx, f in enumerate(files):
        logger.info(f'File {idx} of {num_files}: {f}')
        im_path = os.path.join(folder_to_run, f)
        im_loc = osp.dirname(im_path)
        im_name = im_path.rsplit('/', 1)[-1]
        if result_path is None:
            result_path = im_loc
            im_path_out = osp.join(result_path, im_name.rsplit('.', 1)[-2] + '_seg.png')
            im_path_out_bin = osp.join(result_path, im_name.rsplit('.', 1)[-2] + '_bin_seg.png')
        else:
            os.makedirs(result_path, exist_ok=True)
            im_path_out = osp.join(result_path, im_name.rsplit('.', 1)[-2] + '_seg.png')
            im_path_out_bin = osp.join(result_path, im_name.rsplit('.', 1)[-2] + '_bin_seg.png')
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

        full_pred, full_pred_bin = create_pred(model, im_tens, mask, coords_crop, original_sz,
                                               bin_thresh=bin_thresh, tta=tta)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(im_path_out, img_as_ubyte(full_pred))
            imsave(im_path_out_bin, img_as_ubyte(full_pred_bin))


if __name__ == '__main__':
    folder_to_run = '/Users/austin/test_files/eyes'
    output_dir = '/Users/austin/test_files/eyes/out'
    main(folder_to_run, output_dir)
