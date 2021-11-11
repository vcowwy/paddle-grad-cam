import argparse
import cv2
import numpy as np

import paddle
from models.resnet import resnet50

from paddle_grad_cam import GradCAM
from paddle_grad_cam import ScoreCAM
from paddle_grad_cam import GradCAMPlusPlus
from paddle_grad_cam import AblationCAM
from paddle_grad_cam import XGradCAM
from paddle_grad_cam import EigenCAM
from paddle_grad_cam import EigenGradCAM
from paddle_grad_cam import LayerCAM
from paddle_grad_cam import FullGrad
from paddle_grad_cam import GuidedBackpropReLUModel
from paddle_grad_cam.utils.image import show_cam_on_image
from paddle_grad_cam.utils.image import deprocess_image
from paddle_grad_cam.utils.image import preprocess_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true',
                        default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str,
                        default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--aug_smooth',
                        action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth',
                        action='store_true',
                        help='Reduce noise by taking the first principle componenetof cam_weights*activations')
    parser.add_argument('--method', type=str,
                        default='gradcam++',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and paddle.is_compiled_with_cuda()
    if args.use_cuda:
        print('Using GPU for acceleration')
        paddle.set_device("gpu")
    else:
        print('Using CPU for computation')
        paddle.set_device("cpu")

    return args


if __name__ == '__main__':
    args = get_args()
    methods = {'gradcam': GradCAM,
               'scorecam': ScoreCAM,
               'gradcam++': GradCAMPlusPlus,
               'ablationcam': AblationCAM,
               'xgradcam': XGradCAM,
               'eigencam': EigenCAM,
               'eigengradcam': EigenGradCAM,
               'layercam': LayerCAM,
               'fullgrad': FullGrad}

    model = resnet50()
    model_state_dict = paddle.load("./models/resnet50.pdparams")
    model.set_dict(model_state_dict)

    target_layers = model.layer4[-1]

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255

    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    target_category = None
    cam_algorithm = methods[args.method]

    with cam_algorithm(model=model,
                       target_layers=target_layers) as cam:

        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    gb_model = GuidedBackpropReLUModel(model=model)

    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    np.save(f'{args.method}_cam', cam_image)
