from pypdf import PdfReader


reader = PdfReader("Docs/RanFy_Company_Timeline.pdf")
print("Pages:", len(reader.pages))

for page in reader.pages:
    print(page.extract_text())










    


    