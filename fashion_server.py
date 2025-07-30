# fashion_server.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 클래스 라벨
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 모델 정의 (학습 때와 동일해야 함)
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FashionCNN().to(device)
model.load_state_dict(torch.load("fashion_model.pth", map_location=device))
model.eval()

# 전처리 함수
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        tensor = transform_image(image_bytes).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            label = classes[predicted.item()]

        return JSONResponse(content={"label": label})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
