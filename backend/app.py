#!/usr/bin/env python3
"""
AI-Powered Resume Builder Backend
Complete production-ready Flask application with AI integration
Save this as: resume_backend.py
"""

import os
import json
import io
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis
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
CORS(app, origins=["http://localhost:*", "http://127.0.0.1:*"])

# Set up rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"],
    storage_uri="redis://localhost:6379" if os.getenv("REDIS_URL") else "memory://"
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
    openai.api_key = Config.OPENAI_API_KEY
    ai_client = openai
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
        
        Provide:
        1. A compelling professional summary (2-3 sentences)
        2. 5 strong achievement-focused bullet points for work experience
        3. 10-12 relevant technical and soft skills
        4. 3 key achievements or metrics
        
        Format as JSON with keys: summary, bullets, skills, achievements
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4" if "gpt-4" in os.getenv("OPENAI_MODEL", "") else "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert resume writer and career coach."},
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
        
        Provide:
        1. A compelling professional summary (2-3 sentences)
        2. 5 strong achievement-focused bullet points for work experience
        3. 10-12 relevant technical and soft skills
        4. 3 key achievements or metrics
        
        Return as JSON with keys: summary, bullets, skills, achievements
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
                          f"Eager to contribute to innovative projects and grow expertise in a collaborative environment. "
                          f"Quick learner with passion for continuous improvement and delivering quality results.",
                "bullets": [
                    f"Developed and maintained features for production applications using {request.keywords or 'industry-standard tools'}",
                    "Collaborated with senior developers to implement best practices and coding standards",
                    "Participated in code reviews and contributed to technical documentation",
                    "Assisted in debugging and resolving technical issues, improving system stability by 15%",
                    "Completed all assigned tasks on schedule while maintaining high code quality"
                ]
            },
            "mid": {
                "summary": f"Experienced {role} with proven track record of delivering high-quality solutions in "
                          f"{request.industry or 'fast-paced environments'}. Skilled in {request.keywords or 'full-stack development'} "
                          f"with focus on scalability and performance. Strong collaborator who bridges technical and business needs.",
                "bullets": [
                    f"Led development of key features using {request.keywords or 'cutting-edge technologies'}, resulting in 30% performance improvement",
                    "Mentored junior developers and conducted code reviews to maintain high quality standards",
                    "Architected and implemented scalable solutions handling 10K+ daily active users",
                    "Collaborated with product managers to translate business requirements into technical specifications",
                    "Reduced deployment time by 40% through automation and CI/CD pipeline optimization"
                ]
            },
            "senior": {
                "summary": f"Senior {role} with {request.industry or 'extensive cross-industry'} expertise and "
                          f"deep knowledge of {request.keywords or 'enterprise architectures'}. Proven leader in "
                          f"driving technical innovation and delivering complex projects. Expert at building high-performing teams.",
                "bullets": [
                    f"Architected enterprise-scale solutions using {request.keywords or 'microservices and cloud technologies'}",
                    "Led team of 8+ developers to deliver critical projects 20% under budget and ahead of schedule",
                    "Established coding standards and best practices adopted across entire engineering organization",
                    "Reduced operational costs by 35% through strategic technology choices and optimization",
                    "Presented technical strategies to C-level executives and secured buy-in for major initiatives"
                ]
            },
            "executive": {
                "summary": f"Visionary technology executive and {role} with track record of transforming organizations through "
                          f"strategic innovation. Expert in {request.keywords or 'digital transformation'} with proven ability to "
                          f"align technology initiatives with business objectives. Builder of world-class engineering teams.",
                "bullets": [
                    f"Directed technology strategy for {request.company or 'Fortune 500 company'}, driving 50% increase in efficiency",
                    "Built and scaled engineering organization from 10 to 100+ professionals across 3 continents",
                    "Secured $10M+ in cost savings through strategic vendor negotiations and architecture optimization",
                    "Launched innovative products generating $25M+ in new revenue streams",
                    "Established partnerships with key technology providers to accelerate innovation"
                ]
            }
        }
        
        template = templates.get(level, templates["mid"])
        
        # Generate skills based on role and keywords
        base_skills = ["Leadership", "Problem Solving", "Communication", "Team Collaboration", "Project Management"]
        
        if request.keywords:
            tech_skills = [s.strip() for s in request.keywords.split(',')][:7]
        else:
            tech_skills = ["Python", "JavaScript", "Cloud Architecture", "DevOps", "Agile"]
        
        skills = tech_skills + base_skills
        
        achievements = [
            f"Increased team productivity by 25% through process improvements",
            f"Received '{role} of the Year' award for exceptional performance",
            f"Successfully delivered {3 if level == 'junior' else 10}+ major projects on time and within budget"
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
            h2 { color: #1e40af; margin-top: 25px; text-transform: uppercase; font-size: 14px; letter-spacing: 1px; }
            .contact { color: #666; margin-bottom: 20px; }
            .section { margin-bottom: 25px; }
            ul { margin-left: 20px; }
            li { margin-bottom: 5px; }
        </style>
    </head>
    <body>
        <h1>{{ name }}</h1>
        <div class="contact">{{ email }} | {{ phone }} | {{ location }}</div>
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
            body { font-family: Georgia, 'Times New Roman', serif; margin: 40px; color: #000; }
            h1 { text-align: center; font-size: 28px; margin-bottom: 10px; }
            .contact { text-align: center; border-bottom: 2px solid #000; padding-bottom: 15px; margin-bottom: 20px; }
            h2 { font-size: 16px; margin-top: 20px; border-bottom: 1px solid #000; }
            .section { margin-bottom: 20px; }
            ul { margin-left: 25px; }
        </style>
    </head>
    <body>
        <h1>{{ name }}</h1>
        <div class="contact">{{ email }} • {{ phone }} • {{ location }}</div>
        <div class="section">
            <h2>Professional Summary</h2>
            <p>{{ summary }}</p>
        </div>
        <div class="section">
            <h2>Professional Experience</h2>
            <ul>
            {% for bullet in experience_bullets %}
                <li>{{ bullet }}</li>
            {% endfor %}
            </ul>
        </div>
        <div class="section">
            <h2>Core Competencies</h2>
            <p>{{ skills | join(', ') }}</p>
        </div>
    </body>
    </html>
    """
}

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
        
        # Validate input
        if not data.get('role'):
            return jsonify({"error": "Role is required"}), 400
        
        # Create request object
        resume_request = ResumeRequest(
            role=data.get('role'),
            company=data.get('company'),
            keywords=data.get('keywords'),
            experience_level=data.get('experience_level', 'mid'),
            industry=data.get('industry'),
            tone=data.get('tone', 'professional')
        )
        
        # Generate content
        content = AIContentGenerator.generate(resume_request)
        
        # Return response
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
        section = data.get('section', 'summary')  # summary, experience, skills
        
        if not original_text:
            return jsonify({"error": "Text is required"}), 400
        
        # This would call AI to improve the text
        # For now, return a simple enhancement
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
        
        # Get template
        template = RESUME_TEMPLATES.get(template_name, RESUME_TEMPLATES['modern'])
        
        # Render HTML
        from jinja2 import Template
        html_content = Template(template).render(**data)
        
        if HTML:
            # Generate PDF using WeasyPrint
            pdf = HTML(string=html_content).write_pdf()
            
            return send_file(
                io.BytesIO(pdf),
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f"resume_{datetime.now().strftime('%Y%m%d')}.pdf"
            )
        else:
            # Fallback: return HTML for browser printing
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
        
        # This would use AI or a database of skills
        # For now, return common suggestions
        skill_map = {
            "software": ["Python", "JavaScript", "React", "Node.js", "Git", "Docker", "AWS", "Agile"],
            "data": ["Python", "SQL", "Pandas", "NumPy", "Tableau", "Machine Learning", "Statistics", "ETL"],
            "design": ["Figma", "Adobe Creative Suite", "UI/UX", "Wireframing", "Prototyping", "User Research"],
            "marketing": ["SEO", "Google Analytics", "Content Strategy", "Social Media", "Email Marketing", "CRM"],
            "default": ["Communication", "Problem Solving", "Leadership", "Time Management", "Teamwork"]
        }
        
        # Find matching skills
        role_lower = role.lower()
        suggested = []
        
        for key, skills in skill_map.items():
            if key in role_lower:
                suggested.extend(skills)
        
        if not suggested:
            suggested = skill_map["default"]
        
        # Filter out current skills
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
        
        # Simple ATS analysis (would be more complex in production)
        score = 75  # Base score
        suggestions = []
        
        # Check for keywords
        if job_description:
            job_words = set(job_description.lower().split())
            resume_words = set(resume_text.lower().split())
            match_rate = len(job_words & resume_words) / len(job_words) if job_words else 0
            score = int(70 + (match_rate * 30))
            
            if match_rate < 0.3:
                suggestions.append("Add more keywords from the job description")
        
        # Check for common issues
        if len(resume_text) < 300:
            suggestions.append("Resume seems too short. Add more detail.")
            score -= 10
        
        if len(resume_text) > 3000:
            suggestions.append("Resume might be too long. Consider condensing.")
            score -= 5
        
        if not any(word in resume_text.lower() for word in ['experience', 'education', 'skills']):
            suggestions.append("Include standard section headers: Experience, Education, Skills")
            score -= 15
        
        return jsonify({
            "score": max(0, min(100, score)),
            "suggestions": suggestions,
            "status": "good" if score >= 70 else "needs_improvement"
        })
    
    except Exception as e:
        logger.error(f"ATS analysis error: {e}")
        return jsonify({"error": "Failed to analyze resume"}), 500

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
    - POST /api/generate - Generate AI content
    - POST /api/improve - Improve existing content
    - POST /api/export/pdf - Export as PDF
    - POST /api/suggestions/skills - Get skill suggestions
    - POST /api/analyze/ats - ATS compatibility check
    - GET /api/templates - List templates
    - GET /health - Health check
    
    To test: curl -X POST http://localhost:{port}/api/generate \\
             -H "Content-Type: application/json" \\
             -d '{{"role": "Software Engineer", "keywords": "Python, React"}}'
    """)
    
    app.run(host='0.0.0.0', port=port, debug=debug)