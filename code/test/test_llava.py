# from PIL import Image
# import requests
# from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM
# import torch

# # model = LlavaForConditionalGeneration.from_pretrained()
# # processor = AutoProcessor.from_pretrained("../../_weight/llava")

# model_id = "../../_weight/llava-v1.5-7b"

# prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
# image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
 
 
 
#  llava-1.5-7b-hf
 
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, max_length=30)
outputs = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(outputs)