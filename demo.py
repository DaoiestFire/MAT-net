import yaml
import numpy as np
from argparse import ArgumentParser

import torch
import imageio
from skimage import img_as_ubyte
from skimage.transform import resize
from scipy.spatial import ConvexHull
from tqdm import tqdm

from modules import build_model
from logger import Logger

import sys
import warnings

warnings.filterwarnings('ignore')


def prepare_model(config_path, checkpoint_path, cpu):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    full_models = build_model(config['model_params'], test=True)
    if not cpu:
        for key in full_models:
            full_models[key].cuda()
    Logger.load_cpk(checkpoint_path, full_models, cpu=cpu)
    for key in full_models:
        full_models[key].eval()
    return full_models


def make_animation(image, video, models, relative=True, cpu=True):
    with torch.no_grad():
        regression_module = models.regression_module
        generator = models.generator
        predictions = []
        source = torch.tensor(image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        if not cpu:
            driving = driving.cuda()

        driving_first = driving[:, :, 0]

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if relative is True:
                transform_params = regression_module.encode_transform_params(driving_first, driving_frame)
            else:
                transform_params = regression_module.encode_transform_params(source, driving_frame)
            grid_dict = regression_module.generate_grid(source, transform_params)
            out = generator(source, grid_dict)
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def animate_image(models, source, driving, init=None, cpu=True):
    with torch.no_grad():
        regression_module = models.regression_module
        generator = models.generator
        source = torch.tensor(source[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        driving = torch.tensor(driving[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if init is not None:
            init = torch.tensor(init[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
            driving = driving.cuda()
            if init is not None:
                init = init.cuda()
        if init is not None:
            transform_params = regression_module.encode_transform_params(init, driving)
        else:
            transform_params = regression_module.encode_transform_params(source, driving)
        grid_dict = regression_module.generate_grid(source, transform_params)
        out = generator(source, grid_dict)
        return np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]


def find_best_frame(source, driving, cpu=False):
    print('start to find best frame')
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm = float('inf')
    frame_num = 0
    for index, image in enumerate(driving):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = index
    return frame_num


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", required=True, help="path to checkpoint to restore")

    parser.add_argument("--source", required=True, help="path to source image or video")
    parser.add_argument("--driving", required=True, help="path to driving image or video")
    parser.add_argument("--result", required=True, help="path to output")

    parser.add_argument("--relative", default=True, help="use relative or absolute keypoint coordinates")
    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most aligned with source. (Only for faces, "
                             "requires face_alignment lib)")
    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None, help="Set frame to start from.")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument("--image", action="store_true", help='animate image')
    parser.add_argument("--init", default=None, type=str, help='required path to init image when relative is True')

    opt = parser.parse_args()

    # prepare data
    model_dict = prepare_model(opt.config, opt.checkpoint, opt.cpu)
    source_image = imageio.get_reader(opt.source).get_next_data() \
        if opt.source.endswith('mp4') else imageio.imread(opt.source)
    source_image = resize(source_image, (256, 256))[..., :3]
    if opt.image is True:
        print('animate an image...')
        driving_image = imageio.imread(opt.driving)
        driving_image = resize(driving_image, (256, 256))[..., :3]
        if opt.relative is True:
            assert opt.init is not None, "please provide init image path"
            init_image = imageio.imread(opt.init)
            init_image = resize(init_image, (256, 256))[..., :3]
        else:
            init_image = None
        image_animated = animate_image(model_dict, source_image, driving_image, init_image, cpu=opt.cpu)
        imageio.imsave(opt.result, img_as_ubyte(image_animated))
        sys.exit()

    reader = imageio.get_reader(opt.driving)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    if opt.find_best_frame or opt.best_frame is not None:
        i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, driving_video, cpu=opt.cpu)
        print("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i + 1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, model_dict, relative=opt.relative,
                                             cpu=opt.cpu)
        predictions_backward = make_animation(source_image, driving_backward, model_dict, relative=opt.relative,
                                              cpu=opt.cpu)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, model_dict, relative=opt.relative, cpu=opt.cpu)
    imageio.mimsave(opt.result, [img_as_ubyte(frame) for frame in predictions], fps=fps)
