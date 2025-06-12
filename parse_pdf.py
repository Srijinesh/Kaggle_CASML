from PyPDF2 import PdfReader
from tqdm import tqdm
reader = PdfReader("Data/Psychology2e_WEB.pdf")

text = ""
for page in tqdm(reader.pages):
    text += page.extract_text() + "\n"

with open("Data/Psychology2e_WEB.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("PDF text extraction complete. Text saved to 'Data/Psychology2e_WEB.txt'.")
