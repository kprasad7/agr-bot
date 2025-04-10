from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from prompt_templates import build_interactive_prompt_with_context
from rag_vector_store import get_vector_context_from_query
import boto3, os, json

# Load env vars
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# AWS clients
bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))
translate = boto3.client("translate", region_name=os.getenv("AWS_REGION"))
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

# Session state
user_state = {
    "language": "en",
    "conversation": [],
    "context": {
        "district": None,
        "mandal": None,
        "village": None,
        "crop": None
    }
}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/set-language")
async def set_language(lang: str = Form(...)):
    user_state["language"] = lang
    user_state["conversation"] = []
    user_state["context"] = {"district": None, "mandal": None, "village": None, "crop": None}
    return {"message": f"Language set to {lang}"}

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_handler(request: ChatRequest):
    user_input = request.message.strip()
    lang = user_state["language"]

    # Translate if input is Telugu
    if lang == "te":
        user_input = translate.translate_text(Text=user_input, SourceLanguageCode="auto", TargetLanguageCode="en")["TranslatedText"]

    user_state["conversation"].append({"user": user_input})

    # Handle greeting
    if user_input.lower() in ["hi", "hello"]:
        reply = "Hello! Which district are you in?"
        if lang == "te":
            reply = translate.translate_text(Text=reply, SourceLanguageCode="en", TargetLanguageCode="te")["TranslatedText"]
        return JSONResponse({"reply": reply})

    # Get FAISS-based RAG context
    rag_context = get_vector_context_from_query(user_input)

    # Build prompt for Claude
    history = "\n".join([f"User: {msg['user']}" for msg in user_state["conversation"][-5:]])
    prompt = build_interactive_prompt_with_context(history, user_state["context"], lang, rag_context)

    response = bedrock.invoke_model(
        body=json.dumps({"prompt": prompt, "max_tokens": 1200, "temperature": 0.7}),
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    reply = json.loads(response["body"].read()).get("completion", "Thinking...")

    if lang == "te":
        reply = translate.translate_text(Text=reply, SourceLanguageCode="en", TargetLanguageCode="te")["TranslatedText"]

    user_state["conversation"].append({"bot": reply})
    return JSONResponse({"reply": reply})
