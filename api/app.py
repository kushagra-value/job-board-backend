# Combined app.py script for resume analysis and candidate scraping from Wellfound.
 
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from contextlib import asynccontextmanager
from api.resume_analysis import ResumeAnalyzer
from browser_use import Browser, BrowserConfig, Agent
from langchain_openai import ChatOpenAI
from supabase import create_client, Client
import uuid
from datetime import datetime
import json
import os
import re
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import PyPDF2
import docx
from typing import BinaryIO
from fastapi.middleware.cors import CORSMiddleware


# Global variables for browser and database client
browser = None
db_client = None

# Lifespan event handler to manage resources
@asynccontextmanager
async def lifespan(app: FastAPI):
    global browser, db_client
    browser = Browser(
        config=BrowserConfig(
            cdp_url='http://localhost:9222',
            headless=False,
            chrome_instance_path='C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
        )
    )
    db_client = AsyncIOMotorClient('mongodb+srv://valuebound:E2gfdCBGyPrGy9C@cluster0.d3y7p.mongodb.net/')
    yield
    await browser.close()
    db_client.close()

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)
# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins
    allow_credentials=True,                   # Allow cookies/auth if needed
    allow_methods=["*"],                      # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],                      # Allow all headers
)

# Initialize Supabase client
supabase_url = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
supabase_key = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')
if not supabase_url or not supabase_key:
    raise ValueError("Supabase URL and Key must be set in environment variables")
supabase: Client = create_client(supabase_url, supabase_key)


# Synchronous text extraction functions for uploaded files
def _extract_text_from_pdf_sync(file: BinaryIO) -> str:
    try:
        # Read the file content into bytes
        content = file.read()
        if not content:
            raise ValueError("Empty PDF file")
        # Create a BytesIO object for PyPDF2
        from io import BytesIO
        reader = PyPDF2.PdfReader(BytesIO(content))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        return text
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {str(e)}")

def _extract_text_from_docx_sync(file: BinaryIO) -> str:
    try:
        # Read the file content into bytes
        content = file.read()
        if not content:
            raise ValueError("Empty DOCX file")
        # Create a BytesIO object for python-docx
        from io import BytesIO
        doc = docx.Document(BytesIO(content))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        raise ValueError(f"Error extracting text from DOCX: {str(e)}")

def _extract_text_from_txt_sync(file: BinaryIO) -> str:
    try:
        return file.read().decode('utf-8')
    except Exception as e:
        raise ValueError(f"Error extracting text from TXT: {str(e)}")

def extract_text_sync(upload_file: UploadFile) -> str:
    if upload_file.filename.lower().endswith('.pdf'):
        return _extract_text_from_pdf_sync(upload_file.file)
    elif upload_file.filename.lower().endswith('.docx'):
        return _extract_text_from_docx_sync(upload_file.file)
    elif upload_file.filename.lower().endswith('.txt'):
        return _extract_text_from_txt_sync(upload_file.file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF, DOCX, and TXT are supported.")

# Endpoint to analyze a resume
@app.post("/api/analyze-resume")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(None),
    required_experience: int = Form(0),
    skills: str = Form(None),  # Comma-separated list
):
    try:
        # Validate file
        if not resume.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        # Extract text from uploaded file
        resume_text = extract_text_sync(resume)
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from resume")

        # Process job description and skills
        job_description = job_description.strip() if job_description else None
        skill_list = [skill.strip() for skill in skills.split(",")] if skills else None
        
        # Initialize analyzer and perform analysis
        analyzer = ResumeAnalyzer()  # Assumes API key is in environment
        result = analyzer.analyze_resume(
            resume_text=resume_text,
            job_description=job_description,
            required_experience=required_experience,
            skill_list=skill_list
        )
        
        # Upload resume to Supabase
        unique_filename = f"{uuid.uuid4()}_{resume.filename}"
        resume.file.seek(0)  # Reset file pointer
        # Ensure file is in bytes for Supabase
        file_content = resume.file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Upload raw bytes to Supabase
        supabase.storage.from_("applicants-resume").upload(
            path=unique_filename,
            file=file_content,  # Pass raw bytes instead of BytesIO
            file_options={"content-type": resume.content_type}
        )
        
        public_url = supabase.storage.from_("applicants-resume").get_public_url(unique_filename)
        
        # Store metadata in MongoDB
        db = db_client["applicants"]
        collection = db["applicants-data"]
        applicant_data = {
            "resume_url": public_url,
            "analysis_result": result,
            "uploaded_at": datetime.utcnow()
        }
        await collection.insert_one(applicant_data)
        
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    
# Task description for scraping Wellfound candidates
task_description = (
    "1. Go to https://wellfound.com/recruit/search/new and wait until the search form is fully loaded.\n"
    "2. In the search page, fill in the following fields exactly:\n"
    "   a. Search name: type \"Cyber Security, Network Security, Ethical Hacking, Risk Management, Security Protocols\" into the input labeled 'Search name'.\n"
    "   b. Broadly speaking, what function are you hiring for?: select the radio button with label 'Engineering'. Use the selector input[type=radio][name=\"function\"][value=\"Engineering\"] and wait for the 'Role' options to appear.\n"
    "   c. Role: select the radio button with label 'Software Engineer'. Use the selector input[type=radio][name=\"role\"][value=\"Software Engineer\"]. Wait for 'Location' options to appear.\n"
    "   d. Location: select the radio button with label 'Onsite or Remote'. Use input[type=radio][name=\"locationPreference\"][value=\"Onsite or Remote\"].\n"
    "3. Click the button whose text is 'Create Search' (ensure it's the search form button, not the global search icon), then wait for the results list to load. If the right page appears then ddo not mind filling all fields.\n"
    "4. On the search results page, for each of the first 20 candidate listings:\n"
    "   a. Click on the candidate's name (use the link inside each listing with selector a[data-test=\"candidate-name\"]) to expand the details inline.\n"
    "   b. From the expanded view, capture the full text (including all paragraphs, bullet points, and formatting) of these fields:\n"
    "       - name\n"
    "       - total_experience\n"
    "       - location\n"
    "       - achievements\n"
    "       - experience_section (entire experience details)\n"
    "       - education\n"
    "       - skills\n"
    "       - desired_salary\n"
    "       - desired_role\n"
    "       - remote_work\n"
    "       - desired_location\n"
    "   c. After extraction, scroll as needed to ensure all items load, then move to the next candidate.\n"
    "   d. Repeat for the first 20 listings on page 1.\n"
    "5. Combine all extracted candidate objects into a single JSON array and output it wrapped in ```json ...```.\n"
)

# Function to scrape candidates from Wellfound
async def scrape_wellfound_candidates():
    agent = Agent(
        task=task_description,
        llm=ChatOpenAI(model="gpt-4o"),
        browser=browser
    )
    result = await agent.run()
    result_str = str(result) if not isinstance(result, str) else result
    
    # Prompt to clean and format the scraped data
    prompt = f"""
    The following text contains candidate profiles extracted from multiple pages.
    Your task is to extract all candidate profiles and format them into a single JSON array. Do not summarise the data keep it same.
    Each candidate profile should have the following keys: - name, total_experience, location, achievements,
    experience_section (an object of all the experience details), education, skills, desired_salary, desired_role,
    remote_work, desired_location
    Here is the text:
    {result_str}
    Please provide the combined JSON array of all candidates profiles wrapped in ```json ```.
    """
    llm = ChatOpenAI(model="gpt-4o")
    response = await llm.ainvoke(prompt)
    cleaned_json_str = response.content.strip()
    
    # Extract and parse JSON
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', cleaned_json_str, re.DOTALL)
    json_str = json_match.group(1).strip() if json_match else cleaned_json_str
    try:
        candidates = json.loads(json_str)
        if not isinstance(candidates, list):
            raise ValueError("Expected a list of candidates")
    except Exception as e:
        raise ValueError(f"Error parsing JSON: {e}")
    return candidates

# Dependency to get MongoDB database instance
async def get_db():
    return db_client['wellfound_candidates_db']

# Endpoint to scrape candidates
@app.post("/api/scrape-candidates")
async def scrape_candidates(
    store_in_db: bool = Form(False),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    try:
        candidates = await scrape_wellfound_candidates()
        if store_in_db:
            collection = db['Cyber Security']
            await collection.insert_many(candidates)
        return {"candidates": candidates}
    except Exception as e:
        return {"error": str(e)}, 500

application = app
# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    