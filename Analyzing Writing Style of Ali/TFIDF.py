import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

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

def extract_tfidf_vectors(chunks):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(chunks)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

pdf_paths = [
    r'C:\Users\USER\Desktop\Interim Report\GizliKalmisIstanbulMasali.pdf',
    r'C:\Users\USER\Desktop\Interim Report\Ali Teoman - Karadelik Güncesi.pdf',
    r'C:\Users\USER\Desktop\Interim Report\insansizKonak061607.pdf',
    r'C:\Users\USER\Desktop\Interim Report\Fahim Bey ve Biz.pdf',
    r'C:\Users\USER\Desktop\Interim Report\Ali Nizami Beyin Alafrangaligi ve Seyhligi.pdf',
    r'C:\Users\USER\Desktop\Interim Report\Camlicadaki Enistemiz.pdf',
    r'C:\Users\USER\Desktop\Interim Report\Three_novels_of_four_authors\Three_novels_of_four_authors\Ahmet Hamdi Tanpinar\Huzur.pdf',
    r'C:\Users\USER\Desktop\Interim Report\Three_novels_of_four_authors\Three_novels_of_four_authors\Ahmet Hamdi Tanpinar\Mahur Beste.pdf',
    r'C:\Users\USER\Desktop\Interim Report\Three_novels_of_four_authors\Three_novels_of_four_authors\Ahmet Hamdi Tanpinar\Sahnenin Disindakiler.pdf',
    r'C:\Users\USER\Desktop\Interim Report\Three_novels_of_four_authors\Three_novels_of_four_authors\Halid Ziya Usakligil\Ask-i Memnu.pdf',
    r'C:\Users\USER\Desktop\Interim Report\Three_novels_of_four_authors\Three_novels_of_four_authors\Halid Ziya Usakligil\Kadin Pencesi.pdf',
    r'C:\Users\USER\Desktop\Interim Report\Three_novels_of_four_authors\Three_novels_of_four_authors\Halid Ziya Usakligil\Kirik Hayatlar.pdf',
    r'C:\Users\USER\Desktop\Interim Report\Three_novels_of_four_authors\Three_novels_of_four_authors\Refik Halid Karay\Anahtar.pdf',
    r'C:\Users\USER\Desktop\Interim Report\Three_novels_of_four_authors\Three_novels_of_four_authors\Refik Halid Karay\Bu Bizim Hayatimiz.pdf',
    r'C:\Users\USER\Desktop\Interim Report\Three_novels_of_four_authors\Three_novels_of_four_authors\Refik Halid Karay\Bugunun Saraylisi.pdf'
]

for pdf_path in pdf_paths:
    text = extract_text_from_pdf(pdf_path)
    preprocessed_text = preprocess_text(text)
    chunks = create_text_chunks(preprocessed_text, chunk_size=512)
# Count total number of words
    total_words = len(preprocessed_text)
    print(f"Book: {pdf_path.split(r'/')[-1]} - Total Words: {total_words}")

    # Extract TF-IDF vectors
    tfidf_matrix, feature_names = extract_tfidf_vectors(chunks)

    # Display TF-IDF vectors' shape and feature names
    book_name = pdf_path.split('\\')[-1]
    print(f"Book: {book_name} - TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print("Feature Names:", feature_names)
    total_chunks = len(chunks)
    print(f"Book: {book_name} - Total Chunks: {total_chunks}")
    for i, chunk in enumerate(chunks, start=1):
        print(f"Book: {book_name} - Chunk {i} - Length: {len(chunk.split())} words")
        # print(chunk)
        # print("----") 