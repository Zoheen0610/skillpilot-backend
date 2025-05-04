from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
from dotenv import load_dotenv
import pymongo
from io import BytesIO
from pdfminer.high_level import extract_text
import warnings
from resume_generator import ResumeData, generate_resume, analyze_resume
import uuid
import uvicorn
import google.generativeai as genai

warnings.filterwarnings('ignore', category=UserWarning)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)
g_model = genai.GenerativeModel("gemini-1.5-flash")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000"], # FrontendURL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Career name mapping
CAREER_MAPPING = {
    "Software Development and Engineering": "Software Development and Engineering",
    "Artificial Intelligence": "Artificial Intelligence",
    "Development": "Development",
    "Security": "Cybersecurity",
    "Data Science": "Data Science",
    "User Experience (UX) and User Interface (UI) Design": "UI/UX Design"
}


MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI not found in environment variables")

print(f"Connecting to MongoDB...")
DB_NAME = "career-predictor"
COLLECTION_NAME = "career_insights"

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


try:
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
    print(f"Available collections: {db.list_collection_names()}")
    sample_doc = collection.find_one()
    if sample_doc:
        print("MongoDB connection successful. Found sample document.")
    else:
        print("MongoDB connection successful but no documents found.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    raise e

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Loadingg model
model_path = os.path.join(BASE_DIR, "ML", "svm_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "ML", "vectorizer.pkl")


try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print(f"ML models loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Vectorizer type: {type(vectorizer)}")
except Exception as e:
    print(f"Error loading ML models: {str(e)}")
    raise e

class SkillsInput(BaseModel):
    skills: str

@app.post("/predict")
async def predict_career(skills_input: SkillsInput):
    try:
        print(f"Received skills input: {skills_input.skills}")
        
        try:
            transformed_data = vectorizer.transform([skills_input.skills.lower()])
            print(f"Data transformed successfully")
            prediction = model.predict(transformed_data)[0]
            print(f"Raw prediction: {prediction}")
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            print(f"Input type: {type(skills_input.skills)}")
            print(f"Input value: {skills_input.skills}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during prediction: {str(e)}"
            )
        
        mapped_career = CAREER_MAPPING.get(prediction, prediction)
        print(f"Mapped career: {mapped_career}")
        
        try:
            career_insights = collection.find_one({"Career": mapped_career})
            print(f"Found career insights: {career_insights}")
        except Exception as e:
            print(f"Error querying MongoDB: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error querying database: {str(e)}"
            )
        
        if career_insights:
            career_insights.pop('_id', None)
            
            try:
                response_data = {
                    "prediction": mapped_career,
                    "insights": {
                        "Career": career_insights["Career"],
                        "US_Avg_Salary": career_insights["US Avg Salary"],
                        "India_Avg_Salary": career_insights["India Avg Salary"],
                        "Entry_Level_Salary_India": career_insights["Entry Level Salary India"],
                        "Mid_Level_Salary_India": career_insights["Mid Level Salary India"],
                        "High_Level_Salary_India": career_insights["High Level Salary India"],
                        "Typical_Degrees": career_insights["Typical Degrees"],
                        "In_Demand_Skills": career_insights["In-Demand Skills"],
                        "AI_Tools": career_insights["AI Tools"],
                        "Demand_in_India": career_insights["Demand in India"],
                        "Growth_in_India": career_insights["Growth in India"],
                        "Top_Companies": career_insights["Top Companies"]
                    }
                }
                print(f"Sending response: {response_data}")
                return response_data
            except KeyError as e:
                print(f"Error accessing career insights fields: {str(e)}")
                print(f"Available fields: {list(career_insights.keys())}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing required field in career insights: {str(e)}"
                )
        else:
            print(f"No insights found for career: {mapped_career}")
            all_careers = [doc["Career"] for doc in collection.find({}, {"Career": 1})]
            print(f"Available careers in database: {all_careers}")
            return {
                "prediction": mapped_career,
                "insights": None
            }
            
    except Exception as e:
        print(f"Unexpected error in predict endpoint: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error args: {e.args}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/careers")
async def get_all_careers():
    try:
        careers = list(collection.find({}, {'_id': 0}))
        return careers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-resume")
async def analyze_resume_endpoint(
    resume: UploadFile = File(...),
    required_skills: str = Form(...)
):
    try:
        content = await resume.read()
        pdf_text = extract_text(BytesIO(content))
        
        analysis_result = analyze_resume(pdf_text, required_skills)
        
        return analysis_result
        
    except Exception as e:
        print(f"Error analyzing resume: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-resume")
async def create_resume(data: ResumeData):
    try:
        result = generate_resume(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

conversation_history = {}

class ChatInput(BaseModel):
    query: str
    session_id: str

@app.post("/chat")
async def chat_with_gemini(input: ChatInput):
    try:
        # If session_id doesn't exist, generate a new one
        session_id = input.session_id if input.session_id else str(uuid.uuid4())

        # If the session ID doesn't have a conversation history, create one
        if session_id not in conversation_history:
            conversation_history[session_id] = []

        # Add the current user's query to the conversation history
        conversation_history[session_id].append(f"User: {input.query}")

        # Construct the prompt with the entire conversation history
        prompt = f"""
You are a friendly career counselor chatbot.

- If the user greets (like "hi", "hello"), respond with a friendly greeting.
- If the user asks something vague (like "I need help"), politely ask for more details.
- If the user asks a specific career-related question, give a concise and helpful answer.
- Do not write in bold or markdown formatting.

Conversation History:
{''.join(conversation_history[session_id])}

User input: {input.query}
Response:
"""

        # Generate a response using the model
        response = g_model.generate_content(prompt)

        # Append the bot's response to the conversation history
        conversation_history[session_id].append(f"Bot: {response.text.strip()}")

        # Return the response along with the session ID
        return {"response": response.text.strip(), "session_id": session_id}

    except Exception as e:
        return {"response": "Sorry, something went wrong. Please try again later."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
