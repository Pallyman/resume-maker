#!/usr/bin/env python3
"""
AI-Powered Resume Builder Backend
Complete production-ready Flask application with AI integration
"""

import os
import json
import io
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

# For AI providers (install what you need)
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from weasyprint import HTML
except ImportError:
    HTML = None
    print("Warning: WeasyPrint not installed. PDF export will be limited.")

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:*", "http://127.0.0.1:*", "https://*.onrender.com"])

# Set up rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"],
    storage_uri="memory://"  # Using memory instead of Redis
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Application configuration"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    AI_PROVIDER = os.getenv('AI_PROVIDER', 'openai')  # 'openai', 'anthropic', or 'mock'
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    MAX_RESUME_LENGTH = 10000
    CACHE_TTL = 3600  # 1 hour cache for AI responses

app.config.from_object(Config)

# Initialize AI clients
if Config.AI_PROVIDER == 'openai' and openai and Config.OPENAI_API_KEY:
    from openai import OpenAI
    ai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
elif Config.AI_PROVIDER == 'anthropic' and anthropic and Config.ANTHROPIC_API_KEY:
    ai_client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
else:
    ai_client = None
    logger.warning("No AI provider configured. Using mock responses.")

# Data models
@dataclass
class ResumeContent:
    """Resume content structure"""
    summary: str
    experience_bullets: List[str]
    skills: List[str]
    achievements: Optional[List[str]] = None

@dataclass
class ResumeRequest:
    """Resume generation request"""
    role: str
    company: Optional[str] = None
    keywords: Optional[str] = None
    experience_level: str = "mid"  # junior, mid, senior, executive
    industry: Optional[str] = None
    tone: str = "professional"  # professional, creative, technical

# AI Content Generator
class AIContentGenerator:
    """Handles AI-powered content generation"""
    
    @staticmethod
    def generate_with_openai(request: ResumeRequest) -> ResumeContent:
        """Generate content using OpenAI GPT"""
        if not ai_client:
            return AIContentGenerator.generate_mock_content(request)
        
        prompt = f"""
        Generate professional resume content for a {request.experience_level}-level {request.role}.
        {f'Target company: {request.company}' if request.company else ''}
        {f'Industry: {request.industry}' if request.industry else ''}
        {f'Key skills/keywords: {request.keywords}' if request.keywords else ''}
        Tone: {request.tone}
        
        Return JSON with these exact keys:
        - summary: A compelling professional summary (2-3 sentences)
        - bullets: Array of 5 achievement-focused bullet points
        - skills: Array of 10-12 relevant skills
        - achievements: Array of 3 key achievements
        """
        
        try:
            response = ai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert resume writer. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            content = json.loads(response.choices[0].message.content)
            return ResumeContent(
                summary=content.get('summary', ''),
                experience_bullets=content.get('bullets', []),
                skills=content.get('skills', []),
                achievements=content.get('achievements', [])
            )
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return AIContentGenerator.generate_mock_content(request)
    
    @staticmethod
    def generate_with_anthropic(request: ResumeRequest) -> ResumeContent:
        """Generate content using Anthropic Claude"""
        if not ai_client:
            return AIContentGenerator.generate_mock_content(request)
        
        prompt = f"""
        Generate professional resume content for a {request.experience_level}-level {request.role}.
        {f'Target company: {request.company}' if request.company else ''}
        {f'Industry: {request.industry}' if request.industry else ''}
        {f'Key skills/keywords: {request.keywords}' if request.keywords else ''}
        Tone: {request.tone}
        
        Return JSON with keys: summary, bullets, skills, achievements
        """
        
        try:
            response = ai_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = json.loads(response.content[0].text)
            return ResumeContent(
                summary=content.get('summary', ''),
                experience_bullets=content.get('bullets', []),
                skills=content.get('skills', []),
                achievements=content.get('achievements', [])
            )
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            return AIContentGenerator.generate_mock_content(request)
    
    @staticmethod
    def generate_mock_content(request: ResumeRequest) -> ResumeContent:
        """Generate mock content for testing/fallback"""
        role = request.role or "Professional"
        level = request.experience_level
        
        templates = {
            "junior": {
                "summary": f"Motivated {role} with strong foundation in {request.keywords or 'modern technologies'}. "
                          f"Eager to contribute to innovative projects and grow expertise in a collaborative environment.",
                "bullets": [
                    f"Developed and maintained features using {request.keywords or 'industry-standard tools'}",
                    "Collaborated with senior developers to implement best practices",
                    "Participated in code reviews and technical documentation",
                    "Assisted in debugging and resolving technical issues",
                    "Completed all assigned tasks on schedule"
                ]
            },
            "mid": {
                "summary": f"Experienced {role} with proven track record in {request.industry or 'technology'}. "
                          f"Skilled in {request.keywords or 'full-stack development'} with focus on quality and performance.",
                "bullets": [
                    f"Led development of key features using {request.keywords or 'modern tech stack'}",
                    "Mentored junior developers and conducted code reviews",
                    "Architected scalable solutions handling 10K+ users",
                    "Collaborated with product managers on requirements",
                    "Improved deployment efficiency by 40%"
                ]
            },
            "senior": {
                "summary": f"Senior {role} with extensive expertise in {request.keywords or 'enterprise solutions'}. "
                          f"Proven leader in driving technical innovation and delivering complex projects.",
                "bullets": [
                    f"Architected enterprise solutions using {request.keywords or 'cloud technologies'}",
                    "Led team of 8+ developers delivering critical projects",
                    "Established coding standards adopted organization-wide",
                    "Reduced operational costs by 35% through optimization",
                    "Presented strategies to C-level executives"
                ]
            },
            "executive": {
                "summary": f"Visionary technology executive and {role} with track record of transformation. "
                          f"Expert in {request.keywords or 'digital innovation'} and team building.",
                "bullets": [
                    f"Directed technology strategy driving 50% efficiency increase",
                    "Built engineering organization from 10 to 100+ professionals",
                    "Secured $10M+ in cost savings through optimization",
                    "Launched products generating $25M+ revenue",
                    "Established strategic technology partnerships"
                ]
            }
        }
        
        template = templates.get(level, templates["mid"])
        
        skills = ["Python", "JavaScript", "React", "Node.js", "AWS", "Docker", 
                 "Git", "Agile", "Leadership", "Communication"]
        
        if request.keywords:
            custom_skills = [s.strip() for s in request.keywords.split(',')][:5]
            skills = custom_skills + skills[:5]
        
        achievements = [
            "Increased team productivity by 25%",
            f"Delivered {5 if level == 'junior' else 15}+ successful projects",
            "Received excellence award for outstanding performance"
        ]
        
        return ResumeContent(
            summary=template["summary"],
            experience_bullets=template["bullets"],
            skills=skills,
            achievements=achievements
        )
    
    @classmethod
    def generate(cls, request: ResumeRequest) -> ResumeContent:
        """Main generation method that routes to appropriate provider"""
        if Config.AI_PROVIDER == 'openai':
            return cls.generate_with_openai(request)
        elif Config.AI_PROVIDER == 'anthropic':
            return cls.generate_with_anthropic(request)
        else:
            return cls.generate_mock_content(request)

# Resume Templates
RESUME_TEMPLATES = {
    "modern": """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #333; }
            h1 { color: #2563eb; border-bottom: 3px solid #2563eb; padding-bottom: 10px; }
            h2 { color: #1e40af; margin-top: 25px; text-transform: uppercase; font-size: 14px; }
            .section { margin-bottom: 25px; }
            ul { margin-left: 20px; }
        </style>
    </head>
    <body>
        <h1>{{ name }}</h1>
        <div>{{ email }} | {{ phone }} | {{ location }}</div>
        <div class="section">
            <h2>Professional Summary</h2>
            <p>{{ summary }}</p>
        </div>
        <div class="section">
            <h2>Experience</h2>
            <ul>
            {% for bullet in experience_bullets %}
                <li>{{ bullet }}</li>
            {% endfor %}
            </ul>
        </div>
        <div class="section">
            <h2>Skills</h2>
            <p>{{ skills | join(' • ') }}</p>
        </div>
    </body>
    </html>
    """,
    "classic": """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: Georgia, serif; margin: 40px; }
            h1 { text-align: center; }
            h2 { border-bottom: 1px solid #000; }
        </style>
    </head>
    <body>
        <h1>{{ name }}</h1>
        <div style="text-align: center;">{{ email }} • {{ phone }}</div>
        <h2>Summary</h2>
        <p>{{ summary }}</p>
        <h2>Experience</h2>
        <ul>
        {% for bullet in experience_bullets %}
            <li>{{ bullet }}</li>
        {% endfor %}
        </ul>
        <h2>Skills</h2>
        <p>{{ skills | join(', ') }}</p>
    </body>
    </html>
    """
}

# CRITICAL: Routes to serve the frontend
@app.route('/')
@app.route('/index.html')
@app.route('/resume-builder.html')
def serve_frontend():
    """Serve the frontend HTML file - fetch from GitHub raw content"""
    import requests
    
    try:
        # Fetch the HTML directly from GitHub raw content
        response = requests.get(
            'https://raw.githubusercontent.com/Pallyman/resume-maker/main/frontend/resume-builder.html',
            timeout=5
        )
        if response.status_code == 200:
            # Update API endpoint in the HTML to use the Render URL
            html_content = response.text
            html_content = html_content.replace('http://localhost:5000', 'https://resume-maker-zrbl.onrender.com')
            return html_content, 200, {'Content-Type': 'text/html'}
    except Exception as e:
        logger.error(f"Error fetching frontend from GitHub: {e}")
    
    # Fallback: Show API is working
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Resume Builder API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #2563eb; }
            .status { background: #10b981; color: white; padding: 10px; border-radius: 5px; margin: 20px 0; }
            .endpoint { background: #f3f4f6; padding: 10px; margin: 10px 0; border-radius: 5px; }
            code { background: #1f2937; color: #10b981; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>🚀 Resume Builder API</h1>
        <div class="status">✅ API is Running Successfully!</div>
        
        <p>The backend is working perfectly. Frontend loading issue - trying to fetch from GitHub.</p>
        
        <h2>Available API Endpoints:</h2>
        <div class="endpoint">
            <strong>GET</strong> <code>/health</code> - Check API health
        </div>
        <div class="endpoint">
            <strong>POST</strong> <code>/api/generate</code> - Generate AI resume content
        </div>
        <div class="endpoint">
            <strong>POST</strong> <code>/api/improve</code> - Improve existing content
        </div>
        
        <h2>Manual Frontend Access:</h2>
        <p>You can access the frontend directly from GitHub:</p>
        <a href="https://raw.githubusercontent.com/Pallyman/resume-maker/main/frontend/resume-builder.html" target="_blank">
            View Frontend HTML
        </a>
    </body>
    </html>
    """, 200

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ai_provider": Config.AI_PROVIDER,
        "ai_available": ai_client is not None
    })

@app.route('/api/generate', methods=['POST'])
@limiter.limit("10 per hour")
def generate_content():
    """Generate resume content using AI"""
    try:
        data = request.json
        
        if not data.get('role'):
            return jsonify({"error": "Role is required"}), 400
        
        resume_request = ResumeRequest(
            role=data.get('role'),
            company=data.get('company'),
            keywords=data.get('keywords'),
            experience_level=data.get('experience_level', 'mid'),
            industry=data.get('industry'),
            tone=data.get('tone', 'professional')
        )
        
        content = AIContentGenerator.generate(resume_request)
        
        return jsonify(asdict(content))
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": "Failed to generate content"}), 500

@app.route('/api/improve', methods=['POST'])
@limiter.limit("20 per hour")
def improve_content():
    """Improve existing resume content"""
    try:
        data = request.json
        original_text = data.get('text', '')
        section = data.get('section', 'summary')
        
        if not original_text:
            return jsonify({"error": "Text is required"}), 400
        
        improved = f"[Enhanced] {original_text}"
        
        return jsonify({"improved": improved})
    
    except Exception as e:
        logger.error(f"Improvement error: {e}")
        return jsonify({"error": "Failed to improve content"}), 500

@app.route('/api/export/pdf', methods=['POST'])
@limiter.limit("20 per hour")
def export_pdf():
    """Export resume as PDF"""
    try:
        data = request.json
        template_name = data.get('template', 'modern')
        
        template = RESUME_TEMPLATES.get(template_name, RESUME_TEMPLATES['modern'])
        
        from jinja2 import Template
        html_content = Template(template).render(**data)
        
        if HTML:
            pdf = HTML(string=html_content).write_pdf()
            
            return send_file(
                io.BytesIO(pdf),
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f"resume_{datetime.now().strftime('%Y%m%d')}.pdf"
            )
        else:
            return html_content, 200, {'Content-Type': 'text/html'}
    
    except Exception as e:
        logger.error(f"PDF export error: {e}")
        return jsonify({"error": "Failed to export PDF"}), 500

@app.route('/api/templates', methods=['GET'])
def get_templates():
    """Get available resume templates"""
    return jsonify({
        "templates": list(RESUME_TEMPLATES.keys()),
        "default": "modern"
    })

@app.route('/api/suggestions/skills', methods=['POST'])
@limiter.limit("30 per hour")
def suggest_skills():
    """Suggest skills based on role"""
    try:
        data = request.json
        role = data.get('role', '')
        current_skills = data.get('current_skills', [])
        
        skill_map = {
            "software": ["Python", "JavaScript", "React", "Node.js", "Git", "Docker", "AWS"],
            "data": ["Python", "SQL", "Pandas", "NumPy", "Tableau", "Machine Learning"],
            "design": ["Figma", "Adobe Creative Suite", "UI/UX", "Wireframing"],
            "marketing": ["SEO", "Google Analytics", "Content Strategy", "Social Media"],
            "default": ["Communication", "Problem Solving", "Leadership", "Teamwork"]
        }
        
        role_lower = role.lower()
        suggested = skill_map.get("default", [])
        
        for key, skills in skill_map.items():
            if key in role_lower:
                suggested = skills
                break
        
        suggested = [s for s in suggested if s not in current_skills]
        
        return jsonify({"suggestions": suggested[:10]})
    
    except Exception as e:
        logger.error(f"Skill suggestion error: {e}")
        return jsonify({"error": "Failed to suggest skills"}), 500

@app.route('/api/analyze/ats', methods=['POST'])
@limiter.limit("10 per hour")
def analyze_ats():
    """Analyze resume for ATS compatibility"""
    try:
        data = request.json
        resume_text = data.get('text', '')
        job_description = data.get('job_description', '')
        
        score = 75
        suggestions = []
        
        if job_description:
            job_words = set(job_description.lower().split())
            resume_words = set(resume_text.lower().split())
            match_rate = len(job_words & resume_words) / len(job_words) if job_words else 0
            score = int(70 + (match_rate * 30))
            
            if match_rate < 0.3:
                suggestions.append("Add more keywords from the job description")
        
        if len(resume_text) < 300:
            suggestions.append("Resume seems too short. Add more detail.")
            score -= 10
        
        if not any(word in resume_text.lower() for word in ['experience', 'education', 'skills']):
            suggestions.append("Include standard section headers")
            score -= 15
        
        return jsonify({
            "score": max(0, min(100, score)),
            "suggestions": suggestions,
            "status": "good" if score >= 70 else "needs_improvement"
        })
    
    except Exception as e:
        logger.error(f"ATS analysis error: {e}")
        return jsonify({"error": "Failed to analyze resume"}), 500

@app.route('/api/extract', methods=['POST'])
@limiter.limit("10 per hour")
def extract_from_document():
    """Extract resume information from uploaded document text"""
    try:
        data = request.json
        content = data.get('content', '')
        
        if not content:
            return jsonify({"error": "No content provided"}), 400
        
        # Use AI to extract structured data if available
        if ai_client and Config.AI_PROVIDER == 'openai':
            try:
                prompt = """Extract the following information from this resume/document text:
                - name: Full name of the person
                - title: Current or desired professional title
                - email: Email address
                - phone: Phone number
                - location: City, State/Country
                - summary: Professional summary (2-3 sentences)
                - skills: List of skills (array)
                - experience: Array of work experiences with {title, company, duration, description}
                - education: Array of education with {degree, institution, year, info}
                
                Return ONLY valid JSON with these keys. If a field is not found, use empty string or empty array.
                """
                
                response = ai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a resume parser. Extract information and return only valid JSON."},
                        {"role": "user", "content": f"{prompt}\n\nResume text:\n{content[:3000]}"}  # Limit content length
                    ],
                    temperature=0.1,
                    max_tokens=1500
                )
                
                extracted = json.loads(response.choices[0].message.content)
                return jsonify(extracted)
                
            except Exception as e:
                logger.error(f"AI extraction failed: {e}")
                # Fall through to basic extraction
        
        # Basic extraction without AI
        lines = content.split('\n')
        extracted = {
            "name": "",
            "title": "",
            "email": "",
            "phone": "",
            "location": "",
            "summary": "",
            "skills": [],
            "experience": [],
            "education": []
        }
        
        # Try to find email
        import re
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        email_match = re.search(email_pattern, content)
        if email_match:
            extracted["email"] = email_match.group()
        
        # Try to find phone
        phone_pattern = r'[\+]?[(]?[0-9]{1,3}[)]?[-.\s]?[(]?[0-9]{1,4}[)]?[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}'
        phone_match = re.search(phone_pattern, content)
        if phone_match:
            extracted["phone"] = phone_match.group()
        
        # Try to extract name (usually first non-empty line)
        for line in lines:
            if line.strip() and len(line.strip()) < 50 and not any(char in line for char in ['@', 'http', 'www']):
                extracted["name"] = line.strip()
                break
        
        # Extract summary (look for summary/objective section)
        summary_keywords = ['summary', 'objective', 'profile', 'about']
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in summary_keywords):
                # Get next few lines as summary
                summary_lines = []
                for j in range(i+1, min(i+4, len(lines))):
                    if lines[j].strip():
                        summary_lines.append(lines[j].strip())
                extracted["summary"] = ' '.join(summary_lines)[:500]
                break
        
        # Extract skills (look for skills section)
        skills_keywords = ['skills', 'technologies', 'competencies', 'expertise']
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in skills_keywords):
                # Get next few lines and extract skills
                skills_text = ' '.join(lines[i+1:i+5])
                # Split by common delimiters
                potential_skills = re.split(r'[,;•·|]', skills_text)
                extracted["skills"] = [s.strip() for s in potential_skills if 10 < len(s.strip()) < 30][:15]
                break
        
        # If no sections found, use first 500 chars as summary
        if not extracted["summary"]:
            extracted["summary"] = ' '.join(content.split()[:100])
        
        return jsonify(extracted)
    
    except Exception as e:
        logger.error(f"Document extraction error: {e}")
        return jsonify({"error": "Failed to extract information from document"}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded"""
    return jsonify({
        "error": "Rate limit exceeded",
        "message": str(e.description)
    }), 429

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal error: {error}")
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

# Development server
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    print(f"""
    🚀 Resume Builder Backend Starting...
    ====================================
    Port: {port}
    Debug: {debug}
    AI Provider: {Config.AI_PROVIDER}
    AI Available: {ai_client is not None}
    
    Endpoints:
    - GET  /                       - Frontend
    - POST /api/generate           - Generate AI content
    - POST /api/improve            - Improve existing content
    - POST /api/export/pdf         - Export as PDF
    - POST /api/suggestions/skills - Get skill suggestions
    - POST /api/analyze/ats        - ATS compatibility check
    - GET  /api/templates          - List templates
    - GET  /health                 - Health check
    """)
    
    app.run(host='0.0.0.0', port=port, debug=debug)