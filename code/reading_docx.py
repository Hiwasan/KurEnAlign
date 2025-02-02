from docx import Document

def read_docx(docx_path):
    """
    Reads a .docx file and returns the text content.

    Args:
        docx_path (str): Path to the .docx file.

    Returns:
        str: Text content of the .docx file.
    """
    try:
        doc = Document(docx_path)
        print("Document opened successfully!")
    except Exception as e:
        print(f"Error opening document: {e}")
        return [], []

    source_text = []
    target_text = []
        
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if len(cells) == 3:  # Assuming the table has 3 columns: ID, Mukri, English
                source_text.append(cells[1])
                target_text.append(cells[2])
    return source_text, target_text

# Example usage
if __name__ == "__main__":
    docx_path = r"C:\NLP\Mukri\Mukri.docx"  # Replace with your file path
    source_text, target_text = read_docx(docx_path)

    if source_text and target_text:
        #print extracted texts
        print("Source (Mukri) Text:")
        for text in source_text:
            print(text)
        
        print("\nTarget (English) Text:")
        for text in target_text:
            print(text)
    else:
        print("No data extracted from the document.")