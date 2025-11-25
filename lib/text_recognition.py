import time
from paddleocr import TextRecognition
from PIL import Image, ImageDraw, ImageFont


model = TextRecognition()
start_time = time.time()
output = model.predict(input="lib/output/plate_1_crop.jpg")
end_time = time.time()

# Define high resolution output dimensions
OUTPUT_WIDTH = 800
TEXT_HEIGHT = 150

for res in output:
    res.save_to_json(save_path="./output/res.json")
    
    # Get recognized text and score
    if hasattr(res, 'rec_text'):
        text = res.rec_text
        score = res.rec_score if hasattr(res, 'rec_score') else 0.0
    elif hasattr(res, 'text'):
        text = res.text
        score = res.score if hasattr(res, 'score') else 0.0
    elif isinstance(res, dict) and 'rec_text' in res:
        text = res['rec_text']
        score = res.get('rec_score', 0.0)
    else:
        print(f"Result structure: {res}")
        print(f"Result type: {type(res)}")
        print(f"Result attributes: {dir(res)}")
        text = str(res)
        score = 0.0
    
    print(f"Recognized text: {text}")
    print(f"Confidence score: {score:.4f}")
    
    # Load the original plate image
    plate_img = Image.open("lib/output/plate_1_crop.jpg")
    
    # Resize plate to fit OUTPUT_WIDTH while maintaining aspect ratio
    aspect_ratio = plate_img.height / plate_img.width
    plate_width = OUTPUT_WIDTH
    plate_height = int(OUTPUT_WIDTH * aspect_ratio)
    plate_img = plate_img.resize((plate_width, plate_height), Image.Resampling.LANCZOS)
    
    # Create combined image
    total_height = plate_height + TEXT_HEIGHT
    combined_img = Image.new('RGB', (OUTPUT_WIDTH, total_height), color='white')
    
    # Paste plate at the top
    combined_img.paste(plate_img, (0, 0))
    
    # Draw text and score at the bottom
    draw = ImageDraw.Draw(combined_img)
    
    # Try to use larger fonts
    try:
        text_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 50)
        score_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 30)
    except:
        text_font = ImageFont.load_default()
        score_font = ImageFont.load_default()
    
    # Draw recognized text
    text_bbox = draw.textbbox((0, 0), text, font=text_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (OUTPUT_WIDTH - text_width) // 2
    text_y = plate_height + 20
    draw.text((text_x, text_y), text, fill='black', font=text_font)
    
    # Draw confidence score
    score_text = f"Confidence: {score:.2%}"
    score_bbox = draw.textbbox((0, 0), score_text, font=score_font)
    score_width = score_bbox[2] - score_bbox[0]
    score_x = (OUTPUT_WIDTH - score_width) // 2
    score_y = text_y + 60
    draw.text((score_x, score_y), score_text, fill='gray', font=score_font)
    
    # Save combined result
    combined_img.save("./output/ocr_plate_frame_280.png")


print(f"Execution time: {end_time - start_time:.2f} seconds")