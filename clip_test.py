from transformers import CLIPTextModel, CLIPTokenizer


clip_model_path = "/data/coding/upload-data/data/adrive/CLIP-ViT-H-14-laion2B-s32B-b79K"
clip_model = CLIPTextModel.from_pretrained(clip_model_path)
clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)


sd_model_path = "/data/coding/upload-data/data/adrive/stable-diffusion-2-1-base"
tokenizer = CLIPTokenizer.from_pretrained(sd_model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd_model_path, subfolder="text_encoder")

print("CLIP model loaded successfully")

text = "A photo of a cat"

clip_input = tokenizer(text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",)
clip_output = clip_model(input_ids=clip_input['input_ids'], return_dict=True)

sd_input = tokenizer(text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",)
sd_output = text_encoder(input_ids=sd_input['input_ids'], return_dict=True)

print('hello')