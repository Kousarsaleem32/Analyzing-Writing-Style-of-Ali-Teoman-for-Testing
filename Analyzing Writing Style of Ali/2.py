# import PyPDF2
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# import re
# import nltk
# nltk.download('punkt')

# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with open(pdf_path, 'rb') as file:
#         pdf_reader = PyPDF2.PdfReader(file)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     return text

# def create_text_chunks(text, chunk_size=512):
#     words = text.split()
#     chunks = []
#     current_chunk = []

#     for word in words:
#         current_chunk.append(word)
#         if len(current_chunk) >= chunk_size:
#             chunks.append(' '.join(current_chunk))
#             current_chunk = []

#     if current_chunk:
#         chunks.append(' '.join(current_chunk))

#     return chunks

# pdf_paths = [
#     r'C:\Users\USER\Desktop\Interim Report\GizliKalmisIstanbulMasali.pdf',
#     r'C:\Users\USER\Desktop\Interim Report\Ali Teoman - Karadelik Güncesi.pdf',
#     r'C:\Users\USER\Desktop\Interim Report\insansizKonak061607.pdf'
# ]

# for pdf_path in pdf_paths:
#     text = extract_text_from_pdf(pdf_path)
#     preprocessed_text = preprocess_text(text)
#     chunks = create_text_chunks(preprocessed_text, chunk_size=512)

#     # Display the length and content of each chunk with book identifier
#     book_name = pdf_path.split("\\")[-1]  # Extracting the book name from the path
#     total_chunks = len(chunks)
#     print(f"Book: {book_name} - Total Chunks: {total_chunks}")
#     for i, chunk in enumerate(chunks, start=1):
#         print(f"Book: {book_name} - Chunk {i} - Length: {len(chunk.split())} words")
#         print(chunk)
#         print("----")  # Separate chunks visually

import PyPDF2
import re
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization using NLTK
    words = nltk.word_tokenize(text)
    
    # Stopwords removal
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]
    
    return ' '.join(words)

def create_text_chunks(text, chunk_size=512):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

pdf_paths = [
    r'C:\Users\USER\Desktop\Interim Report\GizliKalmisIstanbulMasali.pdf',
    r'C:\Users\USER\Desktop\Interim Report\Ali Teoman - Karadelik Güncesi.pdf',
    r'C:\Users\USER\Desktop\Interim Report\insansizKonak061607.pdf'
]

for pdf_path in pdf_paths:
    text = extract_text_from_pdf(pdf_path)
    preprocessed_text = preprocess_text(text)
    chunks = create_text_chunks(preprocessed_text, chunk_size=512)

    # Display the length and content of each chunk with book identifier
    book_name = pdf_path.split("\\")[-1]  # Extracting the book name from the path
    total_chunks = len(chunks)
    print(f"Book: {book_name} - Total Chunks: {total_chunks}")
    for i, chunk in enumerate(chunks, start=1):
        print(f"Book: {book_name} - Chunk {i} - Length: {len(chunk.split())} words")
        print(chunk)
        print("----")  # Separate chunks visually
