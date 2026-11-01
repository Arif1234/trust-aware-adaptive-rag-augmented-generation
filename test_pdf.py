import os
from pypdf import PdfReader

folder = "knowledge_base"

for file in os.listdir(folder):

    if file.endswith(".pdf"):

        path = os.path.join(folder, file)

        reader = PdfReader(path)

        text = ""

        for page in reader.pages[:3]:
            text += page.extract_text()

        print("\n")
        print("=" * 50)
        print(file)
        print("=" * 50)

        print(text[:500])