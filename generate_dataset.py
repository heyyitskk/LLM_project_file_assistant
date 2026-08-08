"""
generate_dataset.py
--------------------
One-off utility to expand resumes/ to 30+ diverse resumes and create
job_descriptions/ with 5+ sample JDs, for the RAG project's dataset
requirement. Not part of the graded deliverable pipeline itself
(resume_rag.py / job_matcher.py are) -- just reproducible sample data.

Run once:
    python generate_dataset.py
"""

import random
from pathlib import Path

random.seed(42)

RESUME_DIR = Path("resumes")
JD_DIR = Path("job_descriptions")
RESUME_DIR.mkdir(exist_ok=True)
JD_DIR.mkdir(exist_ok=True)

FIRST_NAMES = [
    "Alex", "Maria", "Wei", "Fatima", "Diego", "Sophie", "Kenji", "Nadia",
    "Liam", "Ingrid", "Samuel", "Yuki", "Elena", "Omar", "Grace", "Raj",
    "Chloe", "Andres", "Mei", "Victor", "Anya", "Tariq", "Isabella", "Noah",
    "Zara",
]
LAST_NAMES = [
    "Chen", "Garcia", "Kowalski", "Hassan", "Rossi", "Dubois", "Tanaka",
    "Ibrahim", "Murphy", "Larsen", "Adeyemi", "Sato", "Petrov", "Farid",
    "Okafor", "Patel", "Bernard", "Silva", "Wong", "Novak", "Kim", "Ali",
    "Romano", "Schmidt", "Nguyen",
]

ROLES = [
    {
        "title": "Backend Engineer",
        "skills": ["Python", "Django", "PostgreSQL", "Redis", "Docker", "AWS", "REST APIs"],
        "summary": "backend engineer focused on building scalable, reliable APIs and services",
    },
    {
        "title": "Data Scientist",
        "skills": ["Python", "Pandas", "scikit-learn", "TensorFlow", "SQL", "Statistics"],
        "summary": "data scientist with a track record of turning data into predictive models",
    },
    {
        "title": "Frontend Developer",
        "skills": ["JavaScript", "TypeScript", "React", "CSS", "HTML", "Jest"],
        "summary": "frontend developer who builds accessible, performant user interfaces",
    },
    {
        "title": "DevOps Engineer",
        "skills": ["Python", "Bash", "Terraform", "AWS", "Kubernetes", "CI/CD", "Docker"],
        "summary": "DevOps engineer automating infrastructure and deployment pipelines",
    },
    {
        "title": "Machine Learning Engineer",
        "skills": ["Python", "PyTorch", "Machine Learning", "NLP", "Docker", "FastAPI"],
        "summary": "machine learning engineer deploying models into production systems",
    },
    {
        "title": "Product Manager",
        "skills": ["Product Strategy", "Roadmapping", "SQL", "Agile", "Stakeholder Management"],
        "summary": "product manager leading cross-functional teams to ship consumer products",
    },
    {
        "title": "QA Engineer",
        "skills": ["Python", "Selenium", "Test Automation", "Jenkins", "API Testing"],
        "summary": "QA engineer building automated test suites to catch regressions early",
    },
    {
        "title": "Data Engineer",
        "skills": ["Python", "SQL", "Airflow", "Spark", "AWS", "ETL"],
        "summary": "data engineer designing pipelines that move and transform large datasets",
    },
    {
        "title": "Mobile Developer",
        "skills": ["Swift", "Kotlin", "React Native", "iOS", "Android", "REST APIs"],
        "summary": "mobile developer shipping native and cross-platform apps",
    },
    {
        "title": "Security Engineer",
        "skills": ["Python", "Networking", "Penetration Testing", "AWS", "SIEM"],
        "summary": "security engineer hardening infrastructure and responding to incidents",
    },
    {
        "title": "UX Designer",
        "skills": ["Figma", "User Research", "Prototyping", "Wireframing", "Design Systems"],
        "summary": "UX designer crafting intuitive experiences backed by user research",
    },
    {
        "title": "Full Stack Developer",
        "skills": ["Python", "JavaScript", "React", "Django", "PostgreSQL", "Docker"],
        "summary": "full stack developer comfortable across the entire web stack",
    },
    {
        "title": "Cloud Architect",
        "skills": ["AWS", "Azure", "Terraform", "Kubernetes", "System Design", "Python"],
        "summary": "cloud architect designing resilient, cost-efficient infrastructure",
    },
    {
        "title": "Marketing Analyst",
        "skills": ["SQL", "Excel", "Tableau", "A/B Testing", "Google Analytics"],
        "summary": "marketing analyst translating campaign data into growth insights",
    },
    {
        "title": "Sales Engineer",
        "skills": ["Python", "SQL", "API Integration", "Salesforce", "Communication"],
        "summary": "sales engineer bridging technical solutions and customer needs",
    },
]

COMPANIES = [
    "Acme Corp", "Beta Inc", "DataWorks", "PixelWorks", "CloudNine Systems",
    "Northwind Apps", "LangTech AI", "Ironclad Systems", "Vertex Labs",
    "BrightPath Analytics", "Skyline Software", "Nimbus Cloud", "Forge Digital",
    "Lumen Technologies", "Anchor Systems", "Meridian Health Tech",
]

UNIVERSITIES = [
    "State University", "Tech University", "Riverside College",
    "Coastal University", "Harborview Business School", "Lakeside University",
    "Midtown University", "Summit Institute of Technology",
    "Fairview University", "Crestwood College",
]

DEGREES = [
    "B.S. Computer Science", "M.S. Computer Science", "B.S. Information Technology",
    "M.S. Data Science", "B.A. Computer Science", "B.S. Software Engineering",
    "MBA", "M.S. Statistics", "B.S. Electrical Engineering",
]


def make_resume(first, last, role, years_exp):
    email = f"{first.lower()}.{last.lower()}@example.com"
    phone = f"(555) {random.randint(100,999)}-{random.randint(1000,9999)}"
    skills = role["skills"]
    company1, company2 = random.sample(COMPANIES, 2)
    degree = random.choice(DEGREES)
    university = random.choice(UNIVERSITIES)
    grad_year = 2024 - years_exp
    start1 = 2024 - min(years_exp, random.randint(2, 4))
    end1 = "Present"
    start2 = grad_year
    end2 = start1

    skill_line = ", ".join(skills)
    primary_skill = skills[0]
    secondary_skill = skills[1] if len(skills) > 1 else skills[0]

    lines = [
        f"{first} {last}",
        role["title"],
        f"Email: {email} | Phone: {phone}",
        "",
        "SUMMARY",
        f"{role['title']} with {years_exp} years of experience as a {role['summary']}.",
        "",
        "SKILLS",
        skill_line,
        "",
        "EXPERIENCE",
        f"{role['title']}, {company1} ({start1} - {end1})",
        f"- Led initiatives using {primary_skill} and {secondary_skill} to improve team output",
        f"- Collaborated across teams to deliver projects on schedule",
    ]

    if years_exp > 2:
        lines += [
            "",
            f"{role['title']} (Junior), {company2} ({start2} - {end2})",
            f"- Built foundational skills in {primary_skill} while contributing to core systems",
        ]

    lines += [
        "",
        "EDUCATION",
        f"{degree}, {university} ({grad_year})",
    ]

    return "\n".join(lines)


def main():
    used_names = set()
    count = 0
    target = 25  # additional resumes on top of the existing 7

    while count < target:
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        name_key = f"{first}_{last}".lower()
        if name_key in used_names:
            continue
        used_names.add(name_key)

        role = random.choice(ROLES)
        years_exp = random.randint(0, 12)

        content = make_resume(first, last, role, years_exp)
        filename = RESUME_DIR / f"resume_{name_key}.txt"
        filename.write_text(content, encoding="utf-8")
        count += 1

    print(f"Generated {count} new resumes in {RESUME_DIR}/")

    # --- Job descriptions ---
    jds = {
        "jd_backend_python.txt": """\
Senior Backend Engineer (Python)

We are looking for a Senior Backend Engineer with strong Python experience
to build and scale our core APIs.

Requirements:
- 5+ years Python experience
- Experience with Django or FastAPI
- Strong knowledge of PostgreSQL and Redis
- Experience with Docker and AWS
- Comfortable designing REST APIs at scale

Nice to have: Kubernetes experience, distributed systems background.
""",
        "jd_data_scientist.txt": """\
Data Scientist

We're hiring a Data Scientist to build predictive models that drive product
decisions.

Requirements:
- 3+ years experience in data science or machine learning
- Strong Python skills (Pandas, scikit-learn)
- Experience with TensorFlow or PyTorch
- Solid grounding in statistics
- SQL proficiency for data extraction

Nice to have: experience presenting insights to non-technical stakeholders.
""",
        "jd_frontend_react.txt": """\
Frontend Developer (React)

Join our team building customer-facing web applications in React.

Requirements:
- 2+ years JavaScript/TypeScript experience
- Strong React experience, including hooks and state management
- Familiarity with testing frameworks (Jest)
- Eye for accessible, responsive UI design

Nice to have: design systems experience, CSS-in-JS familiarity.
""",
        "jd_devops.txt": """\
DevOps Engineer

We need a DevOps Engineer to own our CI/CD pipelines and cloud
infrastructure.

Requirements:
- 4+ years Python or Bash scripting experience
- Hands-on Terraform and AWS experience
- Experience with Kubernetes and Docker
- Track record of building reliable CI/CD pipelines

Nice to have: security hardening experience, on-call incident response.
""",
        "jd_ml_engineer.txt": """\
Machine Learning Engineer (NLP)

We're looking for an ML Engineer to deploy NLP models into production.

Requirements:
- 3+ years Python experience
- Experience with PyTorch and Hugging Face
- Experience deploying models via FastAPI or similar
- Understanding of NLP techniques and embeddings

Nice to have: experience with vector databases and RAG systems.
""",
        "jd_product_manager.txt": """\
Senior Product Manager

We're seeking a Senior Product Manager to own the roadmap for a
consumer-facing product.

Requirements:
- 5+ years product management experience
- Track record of shipping products used by 100K+ users
- Strong SQL skills for data-informed decisions
- Experience running Agile teams

Nice to have: MBA, prior startup experience.
""",
    }

    for filename, content in jds.items():
        (JD_DIR / filename).write_text(content, encoding="utf-8")

    print(f"Generated {len(jds)} job descriptions in {JD_DIR}/")


if __name__ == "__main__":
    main()
