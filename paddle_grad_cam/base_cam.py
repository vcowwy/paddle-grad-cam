import cv2
import numpy as np
import paddle
import ttach as tta

from paddle_grad_cam.utils.svd_on_activations import get_2d_projection


class BaseCAM:

    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 compute_input_gradient=False,
                 uses_gradients=True):

        self.model = model
        self.model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.feature_maps = []

    def hook(self, layer, input, output):
        self.feature_maps.append(output)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        raise Exception('Not Implemented')

    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[int(i), int(target_category[i])]
        return loss

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth=False):
        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       target_category,
                                       activations,
                                       grads)
        weighted_activations = weights[:, :, None, None] * activations

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self, input_tensor, target_category=None, eigen_smooth=False):
        if self.compute_input_gradient:
            input_tensor = paddle.to_tensor(input_tensor,
                                            requires_grad=True)

        self.target_layers.register_forward_post_hook(self.hook)

        output = self.model(input_tensor)

        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.numpy(), axis=-1)
        else:
            assert len(target_category) == input_tensor.size(0)

        if self.uses_gradients:
            self.model.clear_gradients()
            loss = self.get_loss(output, target_category)
            loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   output,
                                                   target_category,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self, input_tensor):
        width = input_tensor.shape[-1]
        height = input_tensor.shape[-2]
        return width, height

    def compute_cam_per_layer(self,
                              input_tensor,
                              output,
                              target_category,
                              eigen_smooth):
        lable_onehot = paddle.nn.functional.one_hot(
            paddle.to_tensor(target_category), num_classes=output.shape[1])

        target = paddle.sum(output * lable_onehot, axis=1)

        gradients = paddle.grad(outputs=[target], inputs=[self.feature_maps[0]])[0]

        activations = self.feature_maps[0].numpy()
        grads = gradients.numpy()

        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []

        cam = self.get_cam_image(input_tensor,
                                 self.target_layers,
                                 target_category,
                                 activations,
                                 grads,
                                 eigen_smooth)
        scaled = self.scale_cam_image(cam, target_size)
        cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    def scale_cam_image(self, cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-07 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def forward_augmentation_smoothing(self,
                                       input_tensor,
                                       target_category=None,
                                       eigen_smooth=False):
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1])]
        )
        cams = []
        for transform in transforms:
            self.feature_maps.clear()
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               target_category,
                               eigen_smooth)

            cam = cam[:, None, :, :]
            cam = paddle.to_tensor(cam)
            cam = transform.deaugment_mask(cam)

            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor,
                 target_category=None,
                 aug_smooth=False,
                 eigen_smooth=False):

        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, target_category, eigen_smooth)

        return self.forward(input_tensor,
                            target_category,
                            eigen_smooth)

    def __del__(self):
        self.feature_maps.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.feature_maps.clear()
        if isinstance(exc_value, IndexError):
            print(f'An exception occurred in CAM with block: {exc_type}. Message: {exc_value}')
            return True
