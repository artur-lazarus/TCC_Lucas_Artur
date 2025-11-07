from ultralytics import YOLO

# Carregue o modelo pré-treinado (versão 's' como exemplo, que é mais rápida)
model = YOLO('keremberke/yolov5s-license-plate') 

# Exporte o modelo para o formato ONNX
# imgsz=640 define o tamanho da imagem de entrada
# opset=12 é uma versão estável do ONNX
model.export(format='onnx', int8=True, simplify=True) 

print("Modelo exportado com sucesso para 'yolov5s.onnx' (ou similar)!")