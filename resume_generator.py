from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import base64
from io import BytesIO

class Experience(BaseModel):
    company: str
    position: str
    dates: str
    description: str

class Education(BaseModel):
    school: str
    degree: str
    dates: str
    description: str

class ResumeData(BaseModel):
    # Required fields
    name: str
    email: str
    phone: str
    location: str
    summary: str
    skills: str
    
    # Optional fields
    experience: Optional[List[Experience]] = None
    education: Optional[List[Education]] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None
    certifications: Optional[str] = None
    awards: Optional[str] = None
    projects: Optional[str] = None
    publications: Optional[str] = None
    volunteer_work: Optional[str] = None
    
    

    class Config:
        from_attributes = True

def create_ats_friendly_resume(data: ResumeData) -> Dict[str, str]:
    """Generate an ATS-friendly resume in PDF format with improved styling."""
    try:
        # Initialize validation flags at the very beginning
        has_valid_experience = False
        has_valid_education = False
        
        if data.experience and len(data.experience) > 0:
            for exp in data.experience:
                if exp.company or exp.position:
                    has_valid_experience = True
                    break
                    
        if data.education and len(data.education) > 0:
            for edu in data.education:
                if edu.school or edu.degree:
                    has_valid_education = True
                    break
        
        # Check if there's any content after summary
        has_content_after_summary = (
            has_valid_experience or
            has_valid_education or
            (data.skills and data.skills.strip()) or
            (data.projects and data.projects.strip()) or
            (data.certifications and data.certifications.strip()) or
            (data.publications and data.publications.strip()) or
            (data.volunteer_work and data.volunteer_work.strip()) or
            (data.awards and data.awards.strip())
        )
        
        # Create a buffer to store the PDF
        buffer = BytesIO()
        
        # Create the PDF document with letter size and margins
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,  # 1 inch margins
            leftMargin=72,
            topMargin=36,    # Smaller top margin
            bottomMargin=72
        )
        
        # Create styles
        styles = getSampleStyleSheet()
        
        # Name style
        name_style = ParagraphStyle(
            'NameStyle',
            parent=styles['Normal'],
            fontSize=16,  # Increased from 14
            leading=18,  # Increased from 16
            alignment=0,  # Left alignment
            spaceAfter=8,  # Increased from 2
            fontName='Helvetica-Bold'
        )
        
        # Contact info style
        contact_style = ParagraphStyle(
            'ContactStyle',
            parent=styles['Normal'],
            fontSize=9,
            leading=10,
            alignment=0,  # Left alignment
            spaceAfter=6
        )
        
        # Summary style
        summary_style = ParagraphStyle(
            'SummaryStyle',
            parent=styles['Normal'],
            fontSize=9,
            leading=11,
            alignment=0,
            spaceAfter=8,
            fontName='Helvetica'
        )
        
        # Section header style
        section_style = ParagraphStyle(
            'SectionStyle',
            parent=styles['Normal'],
            fontSize=10,  # Reduced from 11
            leading=12,
            alignment=0,
            spaceBefore=12,
            spaceAfter=6,
            fontName='Helvetica-Bold',
            textTransform='uppercase'  # Make text uppercase
        )
        
        # Normal text style with indent
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=9,
            leading=11,
            fontName='Helvetica',
            leftIndent=10  # Increased indent for content
        )
        
        # Right-aligned style for dates
        date_style = ParagraphStyle(
            'DateStyle',
            parent=normal_style,
            alignment=2,  # Right alignment
            rightIndent=10,  # Add right indent to move dates left
            leftIndent=0  # Override parent's leftIndent
        )
        
        # Bullet point style with indent
        bullet_style = ParagraphStyle(
            'BulletStyle',
            parent=normal_style,
            leftIndent=22,  # Increased indent to match new content indent
            firstLineIndent=0
        )
        
        # Create the story (content)
        story = []
        
        # Name
        story.append(Paragraph(data.name, name_style))
        story.append(Spacer(1, 4))
        
        # Contact info
        contact_info = [data.email, data.phone, data.location]
        if data.linkedin:
            contact_info.append(f'<link href="{data.linkedin}">LinkedIn</link>')
        if data.github:
            contact_info.append(f'<link href="{data.github}">GitHub</link>')
        if data.portfolio:
            contact_info.append(f'<link href="{data.portfolio}">Portfolio</link>')
        
        contact_text = " | ".join(contact_info)
        story.append(Paragraph(contact_text, contact_style))
        
        # Summary
        story.append(Paragraph(data.summary, summary_style))
        story.append(Spacer(1, 12))
        
        # Experience Section
        if data.experience and len(data.experience) > 0:
            if has_content_after_summary:
                story.append(HRFlowable(
                    width="100%",
                    thickness=1,
                    color=colors.black,
                    spaceBefore=6,
                    spaceAfter=6
                ))
            story.append(Paragraph("PROFESSIONAL EXPERIENCE", section_style))
            for exp in data.experience:
                if exp.company or exp.position:
                    # Company and date in a table
                    exp_header = [
                        [Paragraph(exp.company, normal_style),
                         Paragraph(exp.dates, date_style)]
                    ]
                    exp_table = Table(exp_header, colWidths=[doc.width*0.75, doc.width*0.25])
                    exp_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('LEFTPADDING', (0, 0), (0, 0), 0),
                        ('RIGHTPADDING', (1, 0), (1, 0), 0),
                    ]))
                    story.append(exp_table)
                    
                    # Position
                    story.append(Paragraph(exp.position, normal_style))
                    
                    # Description with bullet points
                    for point in exp.description.split('\n'):
                        if point.strip():
                            story.append(Paragraph(f"• {point.strip()}", bullet_style))
                    story.append(Spacer(1, 6))
        
        # Education Section
        if data.education and len(data.education) > 0:
            if has_valid_experience or has_content_after_summary:
                story.append(HRFlowable(
                    width="100%",
                    thickness=1,
                    color=colors.black,
                    spaceBefore=6,
                    spaceAfter=6
                ))
            story.append(Paragraph("EDUCATION", section_style))
            for edu in data.education:
                if edu.school or edu.degree:
                    # School and date in a table
                    edu_header = [
                        [Paragraph(edu.school, normal_style),
                         Paragraph(edu.dates, date_style)]
                    ]
                    edu_table = Table(edu_header, colWidths=[doc.width*0.75, doc.width*0.25])
                    edu_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('LEFTPADDING', (0, 0), (0, 0), 0),
                        ('RIGHTPADDING', (1, 0), (1, 0), 0),
                    ]))
                    story.append(edu_table)
                    
                    # Degree
                    story.append(Paragraph(edu.degree, normal_style))
                    
                    # Description with bullet points
                    for point in edu.description.split('\n'):
                        if point.strip():
                            story.append(Paragraph(f"• {point.strip()}", bullet_style))
                    story.append(Spacer(1, 6))
        
        # Technical Skills Section
        if data.skills and data.skills.strip():
            skills_list = [skill.strip() for skill in data.skills.split(',') if skill.strip()]
            if skills_list:
                if has_valid_experience or has_valid_education or has_content_after_summary:
                    story.append(HRFlowable(
                        width="100%",
                        thickness=1,
                        color=colors.black,
                        spaceBefore=6,
                        spaceAfter=6
                    ))
                story.append(Paragraph("TECHNICAL SKILLS", section_style))
                skills_text = ', '.join(skills_list)
                story.append(Paragraph(skills_text, normal_style))
        
        # Track if we have any previous sections
        has_previous_section = has_valid_experience or has_valid_education or (data.skills and data.skills.strip())
        
        # Optional Sections
        if data.projects and data.projects.strip():
            project_points = [point.strip() for point in data.projects.split('\n') if point.strip()]
            if project_points:
                if has_previous_section:  # Only add HR if there's previous content
                    story.append(HRFlowable(
                        width="100%",
                        thickness=1,
                        color=colors.black,
                        spaceBefore=6,
                        spaceAfter=6
                    ))
                story.append(Paragraph("PROJECTS", section_style))
                for project in project_points:
                    story.append(Paragraph(f"• {project}", normal_style))
                has_previous_section = True
            
        if data.certifications and data.certifications.strip():
            cert_points = [point.strip() for point in data.certifications.split('\n') if point.strip()]
            if cert_points:
                if has_previous_section:  # Only add HR if there's previous content
                    story.append(HRFlowable(
                        width="100%",
                        thickness=1,
                        color=colors.black,
                        spaceBefore=6,
                        spaceAfter=6
                    ))
                story.append(Paragraph("CERTIFICATIONS", section_style))
                for cert in cert_points:
                    story.append(Paragraph(f"• {cert}", normal_style))
                has_previous_section = True
                    
        if data.publications and data.publications.strip():
            pub_points = [point.strip() for point in data.publications.split('\n') if point.strip()]
            if pub_points:
                if has_previous_section:  # Only add HR if there's previous content
                    story.append(HRFlowable(
                        width="100%",
                        thickness=1,
                        color=colors.black,
                        spaceBefore=6,
                        spaceAfter=6
                    ))
                story.append(Paragraph("PUBLICATIONS", section_style))
                for pub in pub_points:
                    story.append(Paragraph(f"• {pub}", normal_style))
                has_previous_section = True
                    
        if data.volunteer_work and data.volunteer_work.strip():
            volunteer_points = [point.strip() for point in data.volunteer_work.split('\n') if point.strip()]
            if volunteer_points:
                if has_previous_section:  # Only add HR if there's previous content
                    story.append(HRFlowable(
                        width="100%",
                        thickness=1,
                        color=colors.black,
                        spaceBefore=6,
                        spaceAfter=6
                    ))
                story.append(Paragraph("VOLUNTEER WORK", section_style))
                for work in volunteer_points:
                    story.append(Paragraph(f"• {work}", normal_style))
                has_previous_section = True
                    
        if data.awards and data.awards.strip():
            award_points = [point.strip() for point in data.awards.split('\n') if point.strip()]
            if award_points:
                if has_previous_section:  # Only add HR if there's previous content
                    story.append(HRFlowable(
                        width="100%",
                        thickness=1,
                        color=colors.black,
                        spaceBefore=6,
                        spaceAfter=6
                    ))
                story.append(Paragraph("AWARDS & ACHIEVEMENTS", section_style))
                for award in award_points:
                    story.append(Paragraph(f"• {award}", normal_style))
                has_previous_section = True
        
        # Build the PDF
        doc.build(story)
        
        # Get the PDF content
        pdf_content = buffer.getvalue()
        
        # Generate HTML preview
        experience_html = []
        if data.experience and len(data.experience) > 0:
            for exp in data.experience:
                if exp.company or exp.position:
                    bullet_points = []
                    for point in exp.description.split('\n'):
                        if point.strip():
                            bullet_points.append(f'<div style="font-size: 9px; margin-left: 15px;">• {point.strip()}</div>')
                    
                    exp_section = f'''
                        <div style="margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-size: 9px;">{exp.company}</span>
                                <span style="font-size: 9px; margin-right: 0;">{exp.dates}</span>
                            </div>
                            <div style="font-size: 9px; margin-left: 0;">{exp.position}</div>
                            {''.join(bullet_points)}
                        </div>
                    '''
                    experience_html.append(exp_section)
            
        education_html = []
        if data.education and len(data.education) > 0:
            for edu in data.education:
                if edu.school or edu.degree:
                    bullet_points = []
                    for point in edu.description.split('\n'):
                        if point.strip():
                            bullet_points.append(f'<div style="font-size: 9px; margin-left: 15px;">• {point.strip()}</div>')
                    
                    edu_section = f'''
                        <div style="margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-size: 9px;">{edu.school}</span>
                                <span style="font-size: 9px; margin-right: 0;">{edu.dates}</span>
                            </div>
                            <div style="font-size: 9px; margin-left: 0;">{edu.degree}</div>
                            {''.join(bullet_points)}
                        </div>
                    '''
                    education_html.append(edu_section)
            
        # Optional sections HTML
        optional_sections_html = []
        
        # Technical Skills Section
        skills_html = ""
        if data.skills and data.skills.strip():
            skills_list = [skill.strip() for skill in data.skills.split(',') if skill.strip()]
            if skills_list:
                skills_text = ', '.join(skills_list)
                skills_html = f'''
                    <div>
                        <h2 style="font-size: 10px; font-weight: bold; margin: 12px 0 6px 0; text-transform: uppercase;">Technical Skills</h2>
                        <div style="font-size: 9px; margin-left: 15px;">{skills_text}</div>
                    </div>
                '''
        
        # Projects Section
        if data.projects and data.projects.strip():
            project_points = [point.strip() for point in data.projects.split('\n') if point.strip()]
            if project_points:
                optional_sections_html.append(f'''
                    <hr style="border: none; border-top: 1px solid black; margin: 10px 0;">
                    <div>
                        <h2 style="font-size: 10px; font-weight: bold; margin-bottom: 6px; text-transform: uppercase;">Projects</h2>
                        {''.join(f'<div style="font-size: 9px; margin-left: 15px;">• {project}</div>' for project in project_points)}
                    </div>
                ''')
            
        # Certifications Section
        if data.certifications and data.certifications.strip():
            cert_points = [point.strip() for point in data.certifications.split('\n') if point.strip()]
            if cert_points:
                optional_sections_html.append(f'''
                    <hr style="border: none; border-top: 1px solid black; margin: 10px 0;">
                    <div>
                        <h2 style="font-size: 10px; font-weight: bold; margin-bottom: 6px; text-transform: uppercase;">Certifications</h2>
                        {''.join(f'<div style="font-size: 9px; margin-left: 15px;">• {cert}</div>' for cert in cert_points)}
                    </div>
                ''')
            
        # Publications Section
        if data.publications and data.publications.strip():
            pub_points = [point.strip() for point in data.publications.split('\n') if point.strip()]
            if pub_points:
                optional_sections_html.append(f'''
                    <hr style="border: none; border-top: 1px solid black; margin: 10px 0;">
                    <div>
                        <h2 style="font-size: 10px; font-weight: bold; margin-bottom: 6px; text-transform: uppercase;">Publications</h2>
                        {''.join(f'<div style="font-size: 9px; margin-left: 15px;">• {pub}</div>' for pub in pub_points)}
                    </div>
                ''')
            
        # Volunteer Work Section
        if data.volunteer_work and data.volunteer_work.strip():
            volunteer_points = [point.strip() for point in data.volunteer_work.split('\n') if point.strip()]
            if volunteer_points:
                optional_sections_html.append(f'''
                    <hr style="border: none; border-top: 1px solid black; margin: 10px 0;">
                    <div>
                        <h2 style="font-size: 10px; font-weight: bold; margin-bottom: 6px; text-transform: uppercase;">Volunteer Work</h2>
                        {''.join(f'<div style="font-size: 9px; margin-left: 15px;">• {work}</div>' for work in volunteer_points)}
                    </div>
                ''')
            
        # Awards Section
        if data.awards and data.awards.strip():
            award_points = [point.strip() for point in data.awards.split('\n') if point.strip()]
            if award_points:
                optional_sections_html.append(f'''
                    <hr style="border: none; border-top: 1px solid black; margin: 10px 0;">
                    <div>
                        <h2 style="font-size: 10px; font-weight: bold; margin-bottom: 6px; text-transform: uppercase;">Awards & Achievements</h2>
                        {''.join(f'<div style="font-size: 9px; margin-left: 15px;">• {award}</div>' for award in award_points)}
                    </div>
                ''')
        
        html_preview = f'''
        <div style="font-family: Helvetica, Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
            <div>
                <h1 style="font-size: 16px; margin: 0 0 8px 0; font-weight: bold;">{data.name}</h1>
                <p style="font-size: 9px; margin: 0 0 6px 0;">{" | ".join(contact_info)}</p>
                <p style="font-size: 9px; margin: 0 0 12px 0; color: #333;">{data.summary}</p>
            </div>
            
            {experience_html and f"""
            <hr style="border: none; border-top: 1px solid black; margin: 10px 0;">
            <div style="margin-bottom: 15px;">
                <h2 style="font-size: 10px; font-weight: bold; margin: 12px 0 6px 0; text-transform: uppercase;">Professional Experience</h2>
                {''.join(experience_html)}
            </div>
            """ or ''}
            
            {education_html and f"""
            {experience_html and '<hr style="border: none; border-top: 1px solid black; margin: 10px 0;">' or ''}
            <div style="margin-bottom: 15px;">
                <h2 style="font-size: 10px; font-weight: bold; margin: 12px 0 6px 0; text-transform: uppercase;">Education</h2>
                {''.join(education_html)}
            </div>
            """ or ''}
            
            {skills_html and f"""
            {(experience_html or education_html) and '<hr style="border: none; border-top: 1px solid black; margin: 10px 0;">' or ''}
            <div style="margin-bottom: 15px;">
                <h2 style="font-size: 10px; font-weight: bold; margin: 12px 0 6px 0; text-transform: uppercase;">Technical Skills</h2>
                <div style="font-size: 9px; margin-left: 15px;">{', '.join([skill.strip() for skill in data.skills.split(',') if skill.strip()])}</div>
            </div>
            """ or ''}
            
            {''.join(optional_sections_html)}
        </div>
        '''
        
        return {
            "pdf_content": base64.b64encode(pdf_content).decode(),
            "html_preview": html_preview
        }
        
    except Exception as e:
        print(f"Error generating resume: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_resume(data: ResumeData) -> dict:
    try:
        result = create_ats_friendly_resume(data)
        return {
            "pdf_content": result["pdf_content"],
            "html_preview": result["html_preview"],
            "type": "application/pdf"
        }
    except Exception as e:
        print(f"Error in generate_resume: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def calculate_skill_similarity(resume_skills: List[str], required_skills: List[str]) -> float:
    # Convert skills lists to lowercase for better matching
    resume_skills = [skill.lower().strip() for skill in resume_skills]
    required_skills = [skill.lower().strip() for skill in required_skills]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Combine all skills into a single string for each set
    resume_text = " ".join(resume_skills)
    required_text = " ".join(required_skills)
    
    # Create TF-IDF matrix
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, required_text])
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0

def analyze_resume(resume_text: str, required_skills: str) -> dict:
    # Extract skills from resume text
    resume_skills = extract_skills(resume_text)
    
    # Convert required skills string to list
    required_skills_list = [skill.strip() for skill in required_skills.split(',')]
    
    # Calculate similarity score
    similarity_score = calculate_skill_similarity(resume_skills, required_skills_list)
    
    # Find matching and missing skills
    matched_skills = []
    missing_skills = []
    
    for req_skill in required_skills_list:
        skill_found = False
        for resume_skill in resume_skills:
            if req_skill.lower() in resume_skill.lower() or resume_skill.lower() in req_skill.lower():
                matched_skills.append(req_skill)
                skill_found = True
                break
        if not skill_found:
            missing_skills.append(req_skill)
    
    # Generate recommendations based on analysis
    recommendations = generate_recommendations(matched_skills, missing_skills, similarity_score)
    
    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "required_skills": required_skills_list,
        "recommendations": recommendations,
        "similarity_score": similarity_score
    }

def extract_skills(text: str) -> List[str]:
    # Convert text to lowercase for better matching
    text = text.lower()
    
    # Find all mentioned skills in the text
    found_skills = []
    
    # Common variations of skill names
    skill_variations = {
        "javascript": ["javascript", "js", "es6"],
        "python": ["python", "py"],
        "react": ["react", "react.js", "reactjs"],
        "node.js": ["node.js", "nodejs", "node"],
        "java": ["java", "core java", "java se"],
        "c++": ["c++", "cpp"],
        "c#": ["c#", "csharp", "c sharp"],
        "machine learning": ["machine learning", "ml"],
        "artificial intelligence": ["artificial intelligence", "ai"],
        "amazon web services": ["aws", "amazon web services"],
        "microsoft azure": ["azure", "microsoft azure"],
        "google cloud platform": ["gcp", "google cloud", "google cloud platform"],
        "sql": ["sql", "mysql", "postgresql", "oracle sql", "sql server"],
        "nosql": ["nosql", "mongodb", "dynamodb", "cassandra"],
        "docker": ["docker", "containerization"],
        "kubernetes": ["kubernetes", "k8s"],
        "git": ["git", "github", "version control"],
        "ci/cd": ["ci/cd", "continuous integration", "continuous deployment", "jenkins", "gitlab ci"],
        "html": ["html", "html5"],
        "css": ["css", "css3", "scss", "sass"],
        "typescript": ["typescript", "ts"],
        "angular": ["angular", "angularjs"],
        "vue": ["vue", "vuejs", "vue.js"],
        "php": ["php"],
        "ruby": ["ruby", "ruby on rails", "rails"],
        "scala": ["scala"],
        "swift": ["swift", "ios development"],
        "kotlin": ["kotlin", "android development"],
        "tensorflow": ["tensorflow", "tf"],
        "pytorch": ["pytorch", "torch"],
        "pandas": ["pandas", "pd"],
        "numpy": ["numpy", "np"],
        "scikit-learn": ["scikit-learn", "sklearn"],
        "devops": ["devops", "devsecops"],
        "agile": ["agile", "scrum", "kanban"],
        "rest api": ["rest", "rest api", "restful", "api development"],
        "cloud computing": ["cloud computing", "cloud architecture", "cloud services"]
    }
    
    # Search for each skill and its variations in the text
    for main_skill, variations in skill_variations.items():
        for variation in variations:
            if variation in text:
                found_skills.append(main_skill)
                break
    
    return list(set(found_skills))  # Remove duplicates

def generate_recommendations(matched_skills: List[str], missing_skills: List[str], similarity_score: float) -> List[str]:
    recommendations = []
    
    # Calculate match percentage
    total_skills = len(matched_skills) + len(missing_skills)
    match_percentage = (len(matched_skills) / total_skills) * 100 if total_skills > 0 else 0
    
    # Add recommendations based on similarity score
    if similarity_score < 0.3:
        recommendations.append("Your skills profile shows significant gaps with the required skills. Consider taking comprehensive courses or bootcamps.")
    elif similarity_score < 0.6:
        recommendations.append("Your skills profile shows moderate alignment. Focus on acquiring the missing skills through targeted learning.")
    else:
        recommendations.append("Your skills profile shows good alignment. Keep your skills up to date and consider specializing in specific areas.")
    
    # Add specific skill recommendations
    if missing_skills:
        recommendations.append(f"Priority skills to develop: {', '.join(missing_skills[:3])}")
    
    # Add learning recommendations
    if len(missing_skills) > 3:
        recommendations.append("Create practice projects to demonstrate your skills in a portfolio.")
    
    # Add certification recommendations
    if match_percentage < 50:
        recommendations.append("Consider obtaining relevant certifications to strengthen your profile.")
    
    return recommendations 