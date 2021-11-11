import numpy as np

import paddle

from paddle_grad_cam.base_cam import BaseCAM
from paddle_grad_cam.utils.find_layers import find_layer_predicate_recursive
from paddle_grad_cam.utils.svd_on_activations import get_2d_projection


class FullGrad(BaseCAM):

    def __init__(self, model, target_layers,
                 reshape_transform=None):
        def layer_with_2D_bias(layer):
            bias_target_layers = [paddle.nn.Conv2D, paddle.nn.BatchNorm2D]
            if type(layer) in bias_target_layers and layer.bias is not None:
                return True
            return False

        target_layers = find_layer_predicate_recursive(
            model, layer_with_2D_bias)
        super(FullGrad, self).__init__(model,
                                       target_layers,
                                       reshape_transform,
                                       compute_input_gradient=True)
        self.bias_data = [self.get_bias_data(layer).cpu().numpy()
                          for layer in target_layers]

    def get_bias_data(self, layer):
        if isinstance(layer, paddle.nn.BatchNorm2D):
            bias = -(layer.running_mean * layer.weight / paddle.sqrt(layer.running_var + layer.eps)) + layer.bias
            return bias.data
        else:
            return layer.bias.data

    def scale_accross_batch_and_channels(self, tensor, target_size):
        batch_size, channel_size = tensor.shape[:2]
        reshaped_tensor = tensor.reshape(
            batch_size * channel_size, *tensor.shape[2:])
        result = self.scale_cam_image(reshaped_tensor, target_size)
        result = result.reshape(batch_size,
                                channel_size,
                                target_size[1],
                                target_size[0])
        return result

    def compute_cam_per_layer(self,
                              input_tensor,
                              target_category,
                              eigen_smooth):
        input_grad = input_tensor.grad.data.cpu().numpy()
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        cam_per_target_layer = []
        target_size = self.get_target_width_height(input_tensor)

        gradient_multiplied_input = input_grad * input_tensor.data.cpu().numpy()
        gradient_multiplied_input = np.abs(gradient_multiplied_input)
        gradient_multiplied_input = self.scale_accross_batch_and_channels(
            gradient_multiplied_input,
            target_size)
        cam_per_target_layer.append(gradient_multiplied_input)

        assert len(self.bias_data) == len(grads_list)
        for bias, grads in zip(self.bias_data, grads_list):
            bias = bias[None, :, None, None]

            bias_grad = np.abs(bias * grads)
            result = self.scale_accross_batch_and_channels(
                bias_grad, target_size)
            result = np.sum(result, axis=1)
            cam_per_target_layer.append(result[:, None, :])

        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)

        if eigen_smooth:
            cam_per_target_layer = self.scale_accross_batch_and_channels(
                cam_per_target_layer, (target_size[0] // 8, target_size[1] // 8))
            cam_per_target_layer = get_2d_projection(cam_per_target_layer)
            cam_per_target_layer = cam_per_target_layer[:, None, :, :]
            cam_per_target_layer = self.scale_accross_batch_and_channels(
                cam_per_target_layer,  target_size)
        else:
            cam_per_target_layer = np.sum(
                cam_per_target_layer, axis=1)[:, None, :]

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        result = np.sum(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)
