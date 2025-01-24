import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
import torch

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to preprocess text (lowercasing, remove special characters, tokenization)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization using NLTK
    words = nltk.word_tokenize(text)
    
    # Stopwords removal
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]
    
    return ' '.join(words)

# Function to create chunks of text
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

# Function to encode text chunks with BERT
def encode_with_bert(chunks):
    # Initialize BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    encoded_chunks = []

    for chunk in chunks:
        # Tokenize and encode each chunk separately
        tokens = tokenizer.encode_plus(chunk, add_special_tokens=True, max_length=512, padding='max_length', return_tensors='pt', truncation=True)
        
        # Perform encoding with BERT
        with torch.no_grad():
            outputs = model(**tokens)
        
        # Get the embeddings (last hidden states)
        embeddings = outputs.last_hidden_state
        encoded_chunks.append(embeddings)

    return encoded_chunks

# PDF paths
pdf_paths = [
    r'C:\Users\USER\Desktop\Interim Report\GizliKalmisIstanbulMasali.pdf',
    r'C:\Users\USER\Desktop\Interim Report\Ali Teoman - Karadelik Güncesi.pdf',
    r'C:\Users\USER\Desktop\Interim Report\insansizKonak061607.pdf'
]

# Loop through PDFs
for pdf_path in pdf_paths:
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    preprocessed_text = preprocess_text(text)
    
    # Count total number of words
    total_words = len(preprocessed_text.split())
    print(f"Book: {pdf_path.split('/')[-1]} - Total Words: {total_words}")

    # Create text chunks
    chunks = create_text_chunks(preprocessed_text, chunk_size=512)

    # Encode text chunks with BERT
    encoded_chunks = encode_with_bert(chunks)
    
    for i, encoded_chunk in enumerate(encoded_chunks, start=1):
        # Process encoded_chunk as needed (store embeddings, perform analysis, etc.)
        # Example: Print the shape of the encoded chunk
        print(f"Book: {pdf_path.split('/')[-1]} - Chunk {i} - Encoded Chunk Shape: {encoded_chunk.shape}")
