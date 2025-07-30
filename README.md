# FashionMNIST 이미지 분류 웹 애플리케이션

이 프로젝트는 FashionMNIST 데이터를 활용하여 CNN 모델을 학습하고, Gradio를 이용해 웹에서 이미지를 업로드하면 예측 결과를 확인할 수 있는 데모입니다.

## 1. 프로젝트 개요

- PyTorch로 CNN 모델을 구성하고 FashionMNIST 데이터로 학습
- 학습된 모델을 `.pth` 파일로 저장
- FastAPI 서버를 통해 모델 예측 처리
- Gradio를 이용한 사용자 웹 인터페이스 구현

## 2. 파일 구성

- `train_fashion_model.py`: CNN 모델 정의 및 FashionMNIST 학습 코드
- `fashion_model.pth`: 학습 완료된 모델 파일
- `fashion_server.py`: FastAPI 기반 서버 코드 (예측 처리 담당)
- `fashion_client.py`: Gradio 기반 클라이언트 웹앱 코드
- `README.md`: 프로젝트 설명 문서

## 3. 실행 방법

### 3.1 모델 학습

```bash
python train_fashion_model.py
```
