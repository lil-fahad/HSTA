
import os
import io
import json
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPProcessor, CLIPModel

# Load environment variables
load_dotenv()
hf_token = os.getenv('HUGGINGFACE_TOKEN')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HSTA")

# Initialize FastAPI app
app = FastAPI(
    title="HSTA: AI Furniture + AR System",
    description="Upload a home plan (image or PDF) and get smart furniture recommendations ready for AR integration.",
    version="1.0"
)

# Load pretrained CLIP
clip_name = "openai/clip-vit-base-patch32"
logger.info(f"Loading CLIP model: {clip_name}")
clip_processor = CLIPProcessor.from_pretrained(clip_name)
clip_model = CLIPModel.from_pretrained(clip_name)

# Load pretrained Falcon Instruct
llm_name = "tiiuae/falcon-7b-instruct"
logger.info(f"Loading Falcon model: {llm_name}")
tokenizer = AutoTokenizer.from_pretrained(llm_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(llm_name, use_auth_token=hf_token, device_map="auto")

@app.get("/")
async def root():
    return {"message": "HSTA API is live. Use /recommend to get smart furniture recommendations."}

@app.post("/recommend")
async def recommend_furniture(file: UploadFile):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only PNG and JPG images are supported in this version.")
    try:
        image = Image.open(io.BytesIO(await file.read()))
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    candidate_texts = ["modern living room", "classic bedroom", "minimalist office", "industrial kitchen"]
    inputs = clip_processor(text=candidate_texts, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    best_match = candidate_texts[probs.argmax()]

    prompt = (
        f"Given a {best_match}, recommend 3 furniture items. "
        f"Return JSON with fields: name, price, 3D_model_link (glTF/GLB), and shop_link."
    )
    logger.info(f"Prompt to LLM: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=300)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        recommendations = json.loads(generated_text.strip())
    except Exception:
        recommendations = {"raw_output": generated_text, "note": "Could not parse JSON automatically."}

    return JSONResponse(content=recommendations)
