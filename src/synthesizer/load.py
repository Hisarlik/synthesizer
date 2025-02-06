from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_text(textfile_path):
    try:
        # This is a long document we can split up.
        with open(textfile_path) as f:
            textfile = f.read()
    except FileNotFoundError:
        print(f"Error: The file at {textfile_path} was not found.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.create_documents([textfile])
    return texts

if __name__ == "__main__":
    # Example usage
    textfile_path = "examples/acciona/acciona_preprocesado_multimodal.txt"
    texts = load_and_split_text(textfile_path)
    if texts:
        print(texts[0])
        print(texts[1])