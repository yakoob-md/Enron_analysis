import docx
import os

doc_path = 'Enron Email Disclosure Analysis .docx'
output_path = 'scratch/full_report_text.txt'

if os.path.exists(doc_path):
    doc = docx.Document(doc_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(full_text))
    print(f"Successfully wrote to {output_path}")
else:
    print(f"File not found: {doc_path}")
