
# HSTA (Home Smart Tech AR)

🚀 **AI Furniture Recommendation System + AR Integration Ready**

This system:
✅ Accepts uploaded home plans (image files).
✅ Uses AI models (CLIP + Falcon) to analyze the style.
✅ Returns smart furniture recommendations, ready with 3D model links (glTF/GLB) for AR integration.

### How to Run

1️⃣ Create virtual environment:
```bash
python -m venv venv
```

2️⃣ Activate:
```bash
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

3️⃣ Install:
```bash
pip install -r requirements.txt
```

4️⃣ Run server:
```bash
uvicorn main:app --reload
```

5️⃣ Access API:
- Swagger docs: http://127.0.0.1:8000/docs
- Recommendation endpoint: POST /recommend

👑 Built by lil_fahad | Powered by cutting-edge AI 🚀
