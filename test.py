from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("/data/coding/upload-data/data/adrive/CLIP-ViT-H-14-laion2B-s32B-b79K")
processor = CLIPProcessor.from_pretrained("/data/coding/upload-data/data/adrive/CLIP-ViT-H-14-laion2B-s32B-b79K")

url = "/data/coding/train/motion-blurry/LQ/GOPR0372_07_00_000047.png"
image = Image.open(url)
image = image.convert("RGB")
inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities