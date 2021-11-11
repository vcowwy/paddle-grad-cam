import tqdm

import paddle

from paddle_grad_cam.base_cam import BaseCAM


class ScoreCAM(BaseCAM):

    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None):
        super(ScoreCAM, self).__init__(model, target_layers,
                                       reshape_transform=reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        with paddle.no_grad():
            upsample = paddle.nn.UpsamplingBilinear2D(
                size=input_tensor.shape[-2:])
            activation_tensor = paddle.to_tensor(activations)

            upsampled = upsample(activation_tensor)

            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0]
            #maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            maxs = paddle.reshape(maxs, shape=[1, 2048, 1, 1])
            mins = paddle.reshape(mins, shape=[1, 2048, 1, 1])
            upsampled = (upsampled - mins) / (maxs - mins)

            #input_tensors = input_tensor[:, None, :, :] * upsampled[:, :, None, :, :]
            a = paddle.reshape(input_tensor, shape=[1, 1, 3, 224, 224])
            b = paddle.reshape(upsampled, shape=[1, 2048, 1, 224, 224])
            input_tensors = a * b

            if hasattr(self, 'batch_size'):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            for batch_index, tensor in enumerate(input_tensors):
                category = target_category[batch_index]
                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE)):
                    batch = tensor[i:i + BATCH_SIZE, :]
                    outputs = self.model(batch).cpu().numpy()[:, category]
                    scores.extend(outputs)
            scores = paddle.to_tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1])

            weights = paddle.nn.Softmax(axis=-1)(scores).numpy()
            return weights
