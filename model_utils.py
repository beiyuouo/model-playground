import os
import torch
import requests
from torchvision import transforms


def load_model(model_path: str):
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "model.pt")
    model = torch.load(model_path)
    model.eval()
    return model


def get_models():
    models = {}
    for model in os.listdir("models"):
        if model.endswith(".pt") or model.endswith(".pth") or os.path.isdir(model):
            model_name = model.split(".")[0]
            models[model_name] = load_model(os.path.join("models", model))
    return models


def set_device(device: str):
    if device == "cpu":
        return torch.device("cpu")
    else:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")


def model_predict(model, inp, device: str = "cpu"):

    inp = transforms.ToTensor()(inp).unsqueeze(0)
    device = set_device(device)
    model.to(device)
    inp = inp.to(device)

    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
    return confidences


if __name__ == "__main__":
    model_list = ["resnet18", "densenet121", "mobilenet_v2"]
    for model_name in model_list:
        model = torch.hub.load("pytorch/vision:v0.6.0", model_name, pretrained=True)
        torch.save(model, f"models/{model_name}.pt")
