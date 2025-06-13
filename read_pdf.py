import re

text = open("Data/Psychology2e_WEB_pdfminer_trimmed.txt", "r", encoding="utf-8").readlines()
full_text = open("Data/Psychology2e_WEB_pdfminer_trimmed.txt", "r", encoding="utf-8").read()
sub_heading = []
is_heading = False
for line in text:
    if line == "CHAPTER OUTLINE\n":
        is_heading = True
    elif line == "\n":
        is_heading = False
    
    if is_heading and line != "CHAPTER OUTLINE\n":
        sub_heading.append(line)

current_heading = "Introduction to Psychology"
dictionary = {}
# for line in text: