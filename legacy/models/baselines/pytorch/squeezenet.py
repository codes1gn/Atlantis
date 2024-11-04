import torch


def SqueezeNet():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
    # or
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)
    model.eval()
    return model