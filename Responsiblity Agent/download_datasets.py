import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

def create_pdf(filename, title, content_paragraphs):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    Story = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='CenterHeading', alignment=TA_CENTER, fontSize=18, spaceAfter=12))

    Story.append(Paragraph(title, styles["CenterHeading"]))
    Story.append(Spacer(1, 12))

    for p in content_paragraphs:
        if p.startswith("CHAPTER") or p.startswith("Section") or p.startswith("Article"):
             Story.append(Paragraph(p, styles["Heading2"]))
        else:
             Story.append(Paragraph(p, styles["Justify"]))
        Story.append(Spacer(1, 12))

    doc.build(Story)

def generate_datasets():
    datasets = {
        "Dataset/UN_Hate_Speech_Strategy.pdf": (
            "UN Strategy and Plan of Action on Hate Speech",
            [
                "Article 1: Definition of Hate Speech",
                "Hate speech is defined as any kind of communication in speech, writing or behaviour, that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.",
                "Article 2: Tackling Hate Speech",
                "The United Nations system must tackle hate speech at every level. This includes monitoring, analyzing, and addressing the root causes and drivers of hate speech.",
                "Section 1: Online Abuse",
                "Digital platforms must ensure that they do not become conduits for incitement to discrimination, hostility, or violence. Toxic AI outputs that generate slurs or derogatory content regarding religion, ethnicity, or gender violate the fundamental principles of the UN.",
                "Section 2: Protective Measures",
                "Implementation of proactive measures to shield vulnerable populations from targeted hate campaigns, ensuring that technological tools, including AI, are not weaponized."
            ]
        ),
        "Dataset/UNESCO_Countering_Online_Hate_Speech.pdf": (
            "UNESCO: Countering Online Hate Speech",
            [
                "Article 1: Scope of the Problem",
                "Online hate speech is a global phenomenon. It often targets vulnerable groups based on their gender, race, ethnicity, or sexual orientation. Derogatory comments that attack these groups are harmful to social cohesion.",
                "Article 2: The Role of Technology",
                "AI systems must be designed to filter, report, and prevent the dissemination of hate speech. Systems that generate or amplify sexism, racism, or religious intolerance are considered highly toxic and unethical.",
                "Section 1: Educational Interventions",
                "Fostering digital citizenship and media literacy is crucial. Users must be educated on the impact of their online behavior and the consequences of sharing harmful content."
            ]
        ),
        "Dataset/EU_Digital_Services_Act.pdf": (
            "EU Digital Services Act (DSA)",
            [
                "CHAPTER I: General Provisions",
                "Article 1: Subject matter and objective",
                "This Regulation lays down harmonised rules for a safe, predictable, and trusted online environment. It aims to prevent the spread of illegal content online.",
                "CHAPTER II: Liability of providers of intermediary services",
                "Article 3: Definitions",
                "'Illegal content' means any information that, in itself or in relation to an activity, including the sale of products or the provision of services, is not in compliance with Union law or the law of any Member State.",
                "Section 1: Obligations for very large online platforms",
                "Platforms must assess and mitigate systemic risks arising from the design and functioning of their services. This includes the dissemination of illegal content, such as hate speech and child sexual abuse material.",
                "Article 5: Hate Speech and Harassment",
                "The DSA explicitly requires the removal of illegal hate speech and content promoting sexual harassment, racial discrimination, or religious bigotry."
            ]
        ),
        "Dataset/EEOC_Harassment_Guidelines.pdf": (
            "EEOC: Harassment in the Workplace Guidelines",
            [
                "Section 1: What Constitutes Harassment?",
                "Harassment is unwelcome conduct that is based on race, color, religion, sex (including sexual orientation, gender identity, or pregnancy), national origin, older age, disability, or genetic information.",
                "Section 2: Sexual Harassment",
                "Sexual harassment includes unwelcome sexual advances, requests for sexual favors, and other verbal or physical harassment of a sexual nature. Derogatory comments about a person's sex are also prohibited.",
                "Article 1: Employer Liability",
                "Employers are liable for harassment by supervisors that results in a tangible employment action. They are also liable for a hostile work environment created by co-workers if the employer knew or should have known and failed to take prompt action.",
                "Article 2: AI and Workplace Harassment",
                "AI tools used in the workplace must not generate discriminatory or harassing content that could contribute to a hostile work environment based on race, religion, sex, or national origin."
            ]
        ),
        "Dataset/CoE_Combating_Sexism.pdf": (
            "Council of Europe - Preventing and Combating Sexism",
            [
                "CHAPTER I: Understanding Sexism",
                "Article 1: Definition",
                "Sexism is any act, gesture, visual representation, spoken or written words, practice, or behaviour based upon the idea that a person or a group of persons is inferior because of their sex.",
                "CHAPTER II: Areas of Implementation",
                "Section 1: Media, Internet, and Artificial Intelligence",
                "Measures must be taken to prevent sexist language and the objectification of women in the media and online. AI and algorithmic decision-making must not reproduce or exacerbate gender stereotypes and sexism.",
                "Article 2: Sexual Harassment",
                "Sexual harassment is a form of gender-based violence and extreme sexism. AI systems that generate derogatory, sexualized, or harassing content targeting individuals based on their sex violate human dignity."
            ]
        ),
        "Dataset/OHCHR_Rabat_Plan.pdf": (
            "OHCHR: Rabat Plan of Action",
            [
                "Article 1: Freedom of Expression vs. Incitement",
                "The Rabat Plan of Action addresses the distinction between freedom of expression and incitement to national, racial, or religious hatred.",
                "Article 2: The Six-Part Threshold Test",
                "To determine if speech crosses the threshold into criminal incitement, six factors must be considered: context, speaker, intent, content and form, extent of the speech act, and likelihood of harm.",
                "Section 1: Application to AI",
                "AI-generated content that promotes religious, ethnic, or national hatred, or that uses derogatory slurs aimed at specific communities, meets the threshold for harm and must be mitigated by responsible AI governance frameworks."
            ]
        )
    }

    for filename, (title, paragraphs) in datasets.items():
        print(f"Generating {filename}...")
        create_pdf(filename, title, paragraphs)
    print("All datasets generated successfully.")

if __name__ == "__main__":
    generate_datasets()
