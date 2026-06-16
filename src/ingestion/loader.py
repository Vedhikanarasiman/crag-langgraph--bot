import fitz  # PyMuPDF
import os

def load_pdfs(folder_path: str) -> list[dict]:
    """
    Loads all PDFs from a folder and extracts text page by page.
    Returns a list of dicts, each containing text + metadata.
    """
    documents = []

    # Loop through every file in the folder
    for filename in os.listdir(folder_path):

        # Only process PDF files
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            print(f"Loading: {filename}")

            # Open the PDF
            pdf = fitz.open(filepath)

            # Loop through each page
            for page_num in range(len(pdf)):
                page = pdf[page_num]

                # Extract raw text from the page
                text = page.get_text()

                # Skip empty pages
                if not text.strip():
                    continue

                # Store text + metadata together
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": filename,
                        "page": page_num + 1  # humans count from 1
                    }
                })

            print(f"Done: {filename} — {len(pdf)} pages")
            pdf.close()

    print(f"\n Total pages loaded: {len(documents)}")
    return documents