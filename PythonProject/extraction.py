import pdfplumber
import os

PDF_PATH = r"C:\Users\sadik\OneDrive\Desktop\lawyerbot_dataset\raw_sources\Custom Integration\S3-S8COMPUTERSCIENCEANDENGINEERINGFINALSYLLABUS.pdf"

OUT_DIR = r"C:\Users\sadik\OneDrive\Desktop\lawyerbot_dataset\extracted_text"
OUT_PATH = os.path.join(OUT_DIR, "S3-S8COMPUTERSCIENCE_AND_ENGINEERINGFINALSYLLABUS.txt")

os.makedirs(OUT_DIR, exist_ok=True)

all_text = []

with pdfplumber.open(PDF_PATH) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_text.append(text)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(all_text))

print("✅ Extracted text saved to:", OUT_PATH)
