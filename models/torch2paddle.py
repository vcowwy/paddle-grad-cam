import torch
import paddle

from torchvision.models import resnet50


# 保存torchvision.models.resnet50的预训练模型
def save_resnet50():
    model = resnet50(pretrained=True)
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, "resnet50.pth")


# 将resnet50 torch的预训练模型转换为resnet50 paddle的预训练模型
def torch2paddle():
    torch_path = "resnet50.pth"
    paddle_path = "resnet50.pdparams"

    torch_state_dict = torch.load(torch_path)
    paddle_state_dict = {}

    fc_names = ["fc"]

    for k in torch_state_dict:
        v = torch_state_dict[k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]

        if any(flag):
            v = v.transpose()

        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")

        if k not in model_state_dict:
            print(k)
        else:
            paddle_state_dict[k] = v

    paddle.save(paddle_state_dict, paddle_path)


if __name__ == "__main__":
    save_resnet50()
    torch2paddle()