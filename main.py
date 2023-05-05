import os
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import gradio as gr

from model_utils import get_models, model_predict
from utils import load_config

MODEL_ROOT = "/model/"

app = FastAPI()
cfg = load_config()
models = get_models()
model_names = [model_name for model_name in models.keys()]

for model_name in model_names:
    io = gr.Interface(
        fn=lambda x: model_predict(models[model_name], x, cfg.device),
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=3),
        title=model_name,
        description=f"Model {model_name} description",
    )
    app = gr.mount_gradio_app(app, io, path=MODEL_ROOT + model_name)


@app.get("/", response_class=HTMLResponse)
def index():
    html_str = """
    <html>
        <head>
            <title>Model Serving</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <!--bootstrap-->
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        </head>
        <body>
            <h1>Model Serving</h1>
            <p>Available models:</p>
            <ul>
    """
    for model_name in model_names:
        html_str += f"<li><a href='{MODEL_ROOT}{model_name}'>{model_name}</a></li>"

    html_str += """
            </ul>
        </body>
    </html>
    """
    return html_str


# io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
# io2 = gr.Interface(lambda x: "Hello, miaowu" + x + "!", "textbox", "textbox")
# app = gr.mount_gradio_app(app, io, path=MODEL_ROOT + "/model1")
# app = gr.mount_gradio_app(app, io2, path=MODEL_ROOT + "/model2")


# Run this from the terminal as you would normally start a FastAPI app: `uvicorn run:app`
# and navigate to http://localhost:8000/gradio in your browser.
