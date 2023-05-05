import os
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader
import gradio as gr

from model_utils import get_models, model_predict
from utils import load_config

# model settings
MODEL_ROOT = "/model/"
cfg = load_config()
models = get_models()
model_names = [
    {"name": model_name, "description": model_name, "url": MODEL_ROOT + model_name}
    for model_name in models.keys()
]

# templates
templates = Jinja2Templates(directory="templates")
jinja_env = Environment(loader=FileSystemLoader("templates"))

app = FastAPI()

for model in model_names:
    model_name = model["name"]
    io = gr.Interface(
        fn=lambda x: model_predict(models[model_name], x, cfg.device),
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=3),
        title=model_name,
        description=f"{model['description']} model trained on ImageNet.",
    )
    app = gr.mount_gradio_app(app, io, path=MODEL_ROOT + model_name)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "models": model_names}
    )
