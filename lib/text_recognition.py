import time
from paddleocr import TextRecognition


model = TextRecognition()
start_time = time.time()
output = model.predict(input="./output/plate_3.jpg")
for res in output:
    res.save_to_json(save_path="./output/res.json")
    res.save_to_img(save_path="./output/res.png")
end_time = time.time()

print(f"Execution time: {end_time - start_time:.2f} seconds")