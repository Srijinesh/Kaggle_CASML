from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from tqdm import tqdm

# high_level import extract_text

# Specify the path to your PDF file
pdf_path = "Data/Psychology2e_WEB.pdf"

rsrc_mgr = PDFResourceManager()
retstr = StringIO()
laparams = LAParams()
device = TextConverter(rsrc_mgr, retstr, laparams=laparams)
interpreter = PDFPageInterpreter(rsrc_mgr, device)

# Open the PDF file
with open(pdf_path, "rb") as fp:
    # Process each page in the PDF

    # The first 18 pages are not needed, so we will skip them
    # References are also not needed. So we will skip these as well
    page_number = 0
    for page in tqdm(PDFPage.get_pages(fp)):
        page_number += 1
        if page_number <= 18 or page_number > 644:
            continue
        interpreter.process_page(page)
    # Get the text from the StringIO object
    text = retstr.getvalue()
# Close the device and StringIO object
device.close()
retstr.close()

# Print or save the extracted text
text = text.replace("\n\nAccess for free at openstax.org", "") 
# Optionally, save the text to a file
with open("Data/Psychology2e_WEB_pdfminer_trimmed.txt", "w", encoding="utf-8") as text_file:
    text_file.write(text)
