# fashion_client.py
import io
import gradio as gr
import requests

def classify_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    files = {'file': ("image.png", buffered.getvalue(), 'image/png')}
    response = requests.post("http://127.0.0.1:8000/predict", files=files)

    if response.status_code == 200:
        return response.json()['label']
    else:
        return "예측 실패"

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Fashion MNIST 분류기"
)

interface.launch()
