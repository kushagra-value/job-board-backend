import os
import json
import argparse
from typing import List, Dict, Any, Optional, Union
import PyPDF2
import docx
import openai
import logging
from dotenv import load_dotenv

class ResumeAnalyzer:
    """
    A class that analyzes resumes against job descriptions using OpenAI's GPT-4o-mini model.
    
    This class provides functionality to:
    1. Parse PDF and DOCX resumes and job descriptions
    2. Extract relevant information using OpenAI's API
    3. Generate structured analysis in JSON format
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the ResumeAnalyzer with optional API key.
        
        Args:
            api_key (Optional[str]): OpenAI API key. If not provided, it will be loaded from environment variables.
        """
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv()
        
        # Set OpenAI API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
            
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Please provide it as an argument or set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        self.logger.info(f"Extracting text from PDF: {file_path}")
        
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        """
        Extract text from a DOCX file.
        
        Args:
            file_path (str): Path to the DOCX file
            
        Returns:
            str: Extracted text from the DOCX
        """
        self.logger.info(f"Extracting text from DOCX: {file_path}")
        
        try:
            doc = docx.Document(file_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from DOCX: {e}")
            raise
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        """
        Extract text from a TXT file.
        
        Args:
            file_path (str): Path to the TXT file
            
        Returns:
            str: Extracted text from the TXT
        """
        self.logger.info(f"Extracting text from TXT: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            self.logger.error(f"Error extracting text from TXT: {e}")
            raise
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from a file based on its extension.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            str: Extracted text from the file
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self._extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return self._extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            return self._extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
    
    def extract_skills_from_job_description(self, job_description: str) -> List[str]:
        """
        Extract key skills from a job description text using OpenAI API.
        
        Args:
            job_description (str): The job description text
            
        Returns:
            List[str]: Top skills extracted from the job description
        """
        self.logger.info("Extracting skills from job description")
        
        prompt = f"""
        Extract the top 5 most important technical and professional skills from this job description.
        Return only a Python list of strings with no additional text or formatting.
        
        JOB DESCRIPTION:
        {job_description}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a skill extraction expert. Extract only the skills from the job description and return them as a Python list."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            # Extract skills from response
            skills_text = response.choices[0].message.content.strip()
            
            # Parse the skills list
            try:
                # Try to parse as a Python list
                skills = eval(skills_text)
                if not isinstance(skills, list):
                    raise ValueError("Response is not a valid list")
                
                # Ensure all items are strings
                skills = [str(skill) for skill in skills]
                
                return skills
            except Exception as e:
                self.logger.error(f"Error parsing skills list: {e}")
                # Fallback parsing if eval fails
                skills = skills_text.replace('[', '').replace(']', '').split(',')
                skills = [skill.strip().strip('"\'') for skill in skills]
                return [skill for skill in skills if skill]
                
        except Exception as e:
            self.logger.error(f"Error extracting skills: {e}")
            # Return empty list in case of API error
            return []
    
    def create_system_prompt(self) -> str:
        """
        Creates the core system prompt for the resume analyzer.
        
        Returns:
            str: The system prompt
        """
        return """You are an advanced AI interviewer that analyzes resumes against job descriptions. 
Your task is to provide a detailed analysis of a candidate's resume in JSON format according to the prompt specifications. 
Focus on extracting accurate information, providing fair assessments, and generating relevant interview questions.
Always ensure your output is well-structured and contains only valid JSON."""
    
    def create_universal_resume_analysis_prompt(
        self, 
        resume_text: str, 
        job_description: str = "", 
        required_experience: int = 0, 
        skill_list: Optional[List[str]] = None
    ) -> str:
        """
        Creates a comprehensive resume analysis prompt for the OpenAI API.
        
        Args:
            resume_text (str): The full text of the candidate's resume
            job_description (str): The job description text for the role
            required_experience (int): The years of experience required for the role
            skill_list (list): Optional list of specific skills to evaluate
            
        Returns:
            str: The formatted prompt for resume analysis
        """
        # Handle case when skill list is not provided
        skill_list_instruction = ""
        if skill_list:
            skill_list_str = ", ".join([f'"{skill}"' for skill in skill_list])
            skill_list_instruction = f"""
    Evaluate the following specific skills from the provided list: [{skill_list_str}].
    """
        else:
            skill_list_instruction = """
    Since no specific skill list was provided, analyze the job description to identify the top 5 skills needed for this role at the required experience level.
    """

        # Create the experience requirement instruction
        exp_instruction = ""
        if required_experience > 0:
            exp_instruction = f"""
    The role requires {required_experience} years of experience. Use this as a benchmark when evaluating the candidate's qualifications and providing scores.
    """

        return f"""Analyze this resume for professional role suitability, focusing on qualifications, skills, and potential concerns:

RESUME: {resume_text}

JOB DESCRIPTION: {job_description}
{exp_instruction}
{skill_list_instruction}

Generate JSON output in this flattened structure:

{{
  "quick_summary": [
    "Concise point 1 about candidate (≤15 words)",
    "Concise point 2 about candidate (≤15 words)",
    "Concise point 3 about candidate (≤15 words)",
    "Concise point 4 about candidate (≤15 words)",
    "Concise point 5 about candidate (≤15 words)"
  ],
  
  "key_projects": [
    {{
      "name": "Project Name",
      "duration": "X months",
      "scope": "team size/org size impacted",
      "contribution": "specific role/responsibility",
      "impact": "measurable outcome"
    }}
  ],
  
  "validated_skills": {{
    "technical": ["verified tool/technology 1", "verified tool/technology 2"],
    "functional": ["domain-specific skill 1", "domain-specific skill 2"],
    "leadership": ["management/strategy skill 1", "management/strategy skill 2"]
  }},
  
  "unverified_skills": ["skill without evidence 1", "skill without evidence 2"],

  "scoring": {{
    "resume_score": 85,
    "knowledge_score": 80,
    "jd_compatibility_score": 75,
    "overall_score": 78
  }},

  "skill_evaluation": {{
    "skills": [
      {{
        "skill_name": "Python",
        "match_score": 90,
        "remark": "Strong evidence of advanced Python usage in multiple projects"
      }},
      {{
        "skill_name": "SQL",
        "match_score": 70,
        "remark": "Basic SQL knowledge demonstrated but lacks advanced application"
      }}
    ],
    "overall_skill_score": 80,
    "top_missing_skills": ["skill from JD not found in resume 1", "skill from JD not found in resume 2"],
    "Overall_remarks": "Concise overall skills conclusion about the candidate's fit for the role",
  }},

  "overall_assessment": {{
    "knowledge_score": 85,
    "communication_score": 75,
    "keywords_matched_score": 75,
    "remarks": "Concise overall conclusion about the candidate's fit for the role",
    "overall_score": 80
  }},

  "green_flags": {{
    "experience_strengths": [
      {{
        "type": "CAREER_PROGRESSION | TENURE | PROMOTIONS",
        "details": "Specific strength details"
      }}
    ],
    "skill_mastery": [
      {{
        "type": "TECHNICAL_DEPTH | DOMAIN_EXPERTISE",
        "details": "Specific skill details"
      }}
    ],
    "achievement_highlights": [
      {{
        "type": "IMPACT | INNOVATION | SCALE",
        "details": "Specific achievement details"
      }}
    ],
    "cultural_fit": [
      {{
        "type": "VALUES_ALIGNMENT | COLLABORATION | LEADERSHIP",
        "details": "Specific cultural fit details"
      }}
    ],
    "certifications": [
      {{
        "type": "TECHNICAL | DOMAIN | LEADERSHIP",
        "details": "Certification details"
      }}
    ],
    "other_strengths": [
      {{
        "type": "UNIQUE_POSITIVE_ATTRIBUTE",
        "details": "Other positive attribute not falling in above categories"
      }}
    ]
  }},
  "red_flags": {{
    "employment_concerns": [
      {{
        "type": "Job Hopping",
        "reason": "3 technical roles less than 18 months - risk for deep expertise",
        "details": "Frontend Developer (8mo), UX Engineer (11mo), Fullstack (14mo)"
      }}
    ],
    "achievement_concerns": [
      {{
        "type": "Metric Gap",
        "reason": "Team leadership claims lack team size/metrics",
        "example": "Led cross-functional team' without scope details"
      }}
    ],
    "skill_concerns": [
      {{
        "type": "Obsolete Skills",
        "reason": "Cloud skills outdated - last AWS project 3 years ago",
        "skill": "AWS",
        "evidence": "Last cloud-related project ended 2021"
      }}
    ],
    "other_concerns": [
      {{
        "type": "Cultural risk",
        "reason": "No collaboration evidence in team-based roles",
        "details": "All projects described as individual contributions"
      }}
    ]
  }},
  "selection_decision": {{
    "selected": true/false,
    "reason": "Precise reasoning for selection/rejection decision based on professional recruiter analysis"
  }}
}}

Analysis Rules:
1. Quick Summary Generation:
   - Create 5-6 bullet points, each ≤15 words
   - Focus on most relevant qualifications for the job description
   - Include mix of experience, skills, achievements and unique selling points
   - Ensure points are specific and evidence-based, not generic
   - Format for easy scanning by hiring managers

2. Scoring System:
   - Resume Score (out of 100): Quality of resume presentation, clarity, quantification of achievements
   - Knowledge Score (out of 100): Technical and domain expertise based on experience and projects
   - JD Compatibility Score (out of 100): Match between resume and job description requirements
   - Overall Score: Weighted average considering JD compatibility (40%), knowledge (30%), resume quality (30%)

3. Experience Analysis:
   - Calculate only full-time professional experience
   - Part-time roles: Count at 50% of full-time equivalent
   - Internships: Include only if post-graduate or 6+ months duration
   - Identify clear progression through role complexity (junior → mid → senior → lead → management)
   - Extract achievements with quantifiable metrics using this format: "Achieved [X%/amount] [improvement/increase/decrease] in [specific metric] by [specific action]"
   - Standardize tenure calculation (e.g., "2 years 3 months" not "27 months")
   - When dates are month/year only, assume employment began on the 1st of the month
   - Compare actual experience against required experience level for the role

4. Skill Evaluation:
   - For each skill listed in skill_list or extracted from JD:
     - Assign score (0-100) based on evidence in resume
     - Provide brief remark explaining score rationale
     - Calculate overall skill score as weighted average (essential skills weighted higher)
   - List skills from JD not found in resume as "top_missing_skills"
   - Score breakdown: 0-30 (mentioned), 31-60 (some experience), 61-85 (proficient), 86-100 (expert)
   - Overall Remarks: Concise conclusion about candidate's overall skill fit for the role based on skills

5. Overall Assessment:
   - Knowledge Score: Based on depth of experience and skill mastery
   - Communication Score: Quality of resume presentation and impact articulation
   - Keywords Matched Score: Key terms appearing in both resume and job description
   - Remarks: Concise conclusion about candidate's fit for the role
   - Overall Score: Comprehensive evaluation considering all factors

6. Project Evaluation:
   - Categorize scope: Individual (1 person) < Team (2-10) < Cross-functional (11-50) < Enterprise (50+)
   - Verify claimed impact against role level and project duration
   - For each project, identify at least one specific contribution and one measurable outcome
   - Flag projects with impact claims disproportionate to role seniority (e.g., entry-level claiming enterprise-wide impact)
   - Require time-bound project descriptions with clear start/end dates or durations

7. Skill Validation:
   - Technical: Must have supporting project/role evidence showing practical application
   - Functional: Match against industry standard requirements; require evidence of practical application
   - Leadership: Require team size/budget/scope mentions to validate management experience
   - Only include skills in "validated_skills" when there is clear evidence of application
   - Categorize skill levels based on evidence: Beginner (mentioned/coursework), Intermediate (1-2 applications), Advanced (3+ applications/leadership)

8. Red Flag Detection:
   - Job Hopping: Flag if a candidate has 3+ distinct roles with durations under 18 months each, excluding cases where the roles represent internal progression within a single company or legitimate transitions (e.g., internships converting to full-time positions)
   - Overlaps: Flag any instance where roles overlap for more than 1 month concurrently
   - Gaps: Flag any unexplained employment gap lasting more than 6 months
   - Skill Mismatch: Flag claims of expert-level proficiency that lack supporting evidence from project work or role responsibilities
   - Downgrade: Flag any move to a less senior role without clear explanation (e.g., Director → Manager)
   - Objective criteria for "vague claims": No specific metrics, no clear scope definition, no description of personal contribution
   - Other Concerns: Include any other red flags not falling into the above categories
   - Format reasons as: [Observable Pattern] + [Specific Risk] + [Decision Impact]
      - Examples for reasons:
        - "5-year gap in technical roles - hard skill currency risk"
        - "Python claims only in education - no production evidence"
        - "Startup experience only - may lack enterprise process knowledge"
        - "Manager title with no reports listed - scope inflation risk"
        - "Certifications without implementation - theoretical knowledge only"
        - "Consistent individual contributor - leadership readiness unclear"
   
   - Include mitigation guidance:
     - "interview_focus": Specific area to probe
     - "development_needed": Required training/certification
     - "verification_required": Documents to request
     - "comparative_risk": How this compares to other candidates

   - Severity tied to role requirements:
     - HIGH: Core requirement deficiency
     - MEDIUM: Secondary skill gap
     - LOW: Nice-to-have missing

    - Follow Question Design rules for red flags related interview questions generation

9. Green Flag Detection:
   - Career Progression: Flag consistent upward mobility with increasing responsibility (clear title progression)
   - Technical Depth: Validate expertise with multiple projects (3+) using same technology stack
   - High Impact: Identify achievements with measurable business impact (must include specific metrics)
   - Cultural Indicators: Note volunteer work, mentoring, or community contributions with specific details
   - Certifications: Highlight role-relevant certifications with practical application evidence
   - Include ONLY strengths with concrete evidence (not aspirational or general statements)
   - Other Strengths: Include any other positive attributes not falling into the above categories

10. Consistency Check:
    - IMPORTANT: Ensure green flags and red flags do not contradict each other. The same attribute cannot be both a strength and a concern.
    - For career trajectory: If "career progression" is listed as a green flag, there should not be "downgrade" as a red flag.
    - For skills: If a skill is listed in "validated_skills", it should not appear in "skill_concerns".
    - For achievements: If listed in "notable_achievements", they should not appear in "achievement_concerns".
    - When in doubt, categorize an element as either a green flag OR a red flag, not both.
    - Review all outputs for logical consistency before finalizing

11. Edge Case Handling:
    - Career Transitions: Consider intentional industry/function changes when evaluating progression
    - Freelance/Consulting: Count consistent client work as stable employment (with evidence of continuing clients)
    - Education Gaps: Do not penalize gaps explained by full-time education
    - Recent Graduates: Adjust expectations for early-career candidates (less than 3 years experience)
    - Founder/Entrepreneur: Evaluate based on company milestones rather than traditional progression
    - Industry-Specific: Adjust expectations for industries with known high turnover (e.g., startups, agencies)
    - Career Breaks: Consider parental leave, health issues, or care responsibilities with appropriate context

12. Missing Information Handling:
    - For missing dates: Note as a data quality issue rather than a red flag
    - For ambiguous titles: Base analysis on responsibilities described rather than title alone
    - For missing metrics: Flag as an achievement concern if senior-level role (5+ years experience)
    - For incomplete employment records: Note limitations in analysis
    - Required fields: If any of these are missing, explicitly note limitation: current role, tenure, and at least one achievement

13. Context-Aware Analysis:
    - Consider industry norms when evaluating tenure (e.g., 2 years in tech startups may be normal)
    - Adjust expectations based on career stage (early/mid/senior/executive)
    - For highly specialized roles, focus on depth rather than breadth of experience
    - Consider geographic context for employment patterns and role expectations
    - Compare experience to typical industry benchmarks rather than absolute standards

14. Selection Decision Guidelines:
    - Selection is a binary decision (true/false) representing whether to advance the candidate
    - The decision should be a professional judgment weighing both qualifications and concerns
    - Consider critical thresholds:
      - TRUE: Minimum overall score of 75+ AND no HIGH severity red flags in core requirements
      - FALSE: Overall score below 65 OR multiple HIGH severity red flags in essential areas
      - BORDERLINE (65-74): Decision based on balance of green flags vs. red flags and market conditions
    - The reason must be precise, specific to this candidate, and reflect professional recruiter judgment
    - Format reason as a clear, objective statement that could be defended to hiring managers
    - Reason should reference specific strengths/concerns from the analysis that drove the decision
    - Decision should consider role requirements, company culture, and market conditions
    - If borderline, err on side of inclusion only if specific green flags outweigh red flags
"""
    
    def analyze_resume(
        self, 
        resume_text: str, 
        job_description: str = "", 
        required_experience: int = 0, 
        skill_list: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a resume against a job description using OpenAI API.
        
        Args:
            resume_text (str): The full text of the candidate's resume
            job_description (str): The job description text for the role
            required_experience (int): The years of experience required for the role
            skill_list (list): Optional list of specific skills to evaluate
            
        Returns:
            dict: JSON result of the resume analysis
        """
        self.logger.info("Starting resume analysis")
        
        # If no skill list is provided but job description is available, extract skills
        if not skill_list and job_description:
            skill_list = self.extract_skills_from_job_description(job_description)
            self.logger.info(f"Extracted skills from job description: {skill_list}")
        
        # Create the system prompt
        system_prompt = self.create_system_prompt()
        
        # Create the analysis prompt
        analysis_prompt = self.create_universal_resume_analysis_prompt(
            resume_text=resume_text,
            job_description=job_description,
            required_experience=required_experience,
            skill_list=skill_list
        )
        
        # Call OpenAI API
        try:
            self.logger.info("Calling OpenAI API for resume analysis")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.5
            )
            
            # Parse the JSON response
            analysis_result = json.loads(response.choices[0].message.content)
            self.logger.info("Successfully received and parsed API response")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in resume analysis: {e}")
            raise
    
    def analyze_resume_from_files(
        self, 
        resume_file_path: str, 
        job_description_file_path: Optional[str] = None, 
        required_experience: int = 0, 
        skill_list: Optional[List[str]] = None,
        output_file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a resume from a file against a job description file.
        
        Args:
            resume_file_path (str): Path to the resume file
            job_description_file_path (Optional[str]): Path to the job description file
            required_experience (int): The years of experience required for the role
            skill_list (Optional[List[str]]): Optional list of specific skills to evaluate
            output_file_path (Optional[str]): Path to save the output JSON file
            
        Returns:
            dict: JSON result of the resume analysis
        """
        self.logger.info(f"Analyzing resume from file: {resume_file_path}")
        
        # Extract text from resume file
        resume_text = self.extract_text_from_file(resume_file_path)
        
        # Extract text from job description file if provided
        job_description = ""
        if job_description_file_path:
            job_description = self.extract_text_from_file(job_description_file_path)
        
        # Analyze the resume
        analysis_result = self.analyze_resume(
            resume_text=resume_text,
            job_description=job_description,
            required_experience=required_experience,
            skill_list=skill_list
        )
        
        # Save the output to a file if requested
        if output_file_path:
            self.logger.info(f"Saving analysis result to: {output_file_path}")
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2)
        
        return analysis_result


def main():
    """
    Main function to run the ResumeAnalyzer from command line.
    """
    parser = argparse.ArgumentParser(description='Analyze a resume against a job description.')
    parser.add_argument('--resume', '-r', required=True, help='Path to the resume file')
    parser.add_argument('--job-description', '-j', help='Path to the job description file')
    parser.add_argument('--experience', '-e', type=int, default=0, help='Required years of experience for the role')
    parser.add_argument('--skills', '-s', nargs='+', help='List of specific skills to evaluate')
    parser.add_argument('--output', '-o', help='Path to save the output JSON file')
    parser.add_argument('--api-key', '-k', help='OpenAI API key')
    
    args = parser.parse_args()
    
    try:
        # Initialize ResumeAnalyzer
        analyzer = ResumeAnalyzer(api_key=args.api_key)
        
        # Analyze the resume
        result = analyzer.analyze_resume_from_files(
            resume_file_path=args.resume,
            job_description_file_path=args.job_description,
            required_experience=args.experience,
            skill_list=args.skills,
            output_file_path=args.output
        )
        
        # Print the result
        print(json.dumps(result, indent=2))
        
        # Print output file path if provided
        if args.output:
            print(f"\nAnalysis saved to: {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0



if __name__ == "__main__":
    exit(main())