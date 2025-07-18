from flask import Flask, render_template, request, send_file
import os
os.environ["USE_TF"] = "0"  # Disable TensorFlow usage in transformers

# Flask app initialization
app = Flask(__name__)

import torch
import spacy
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
import fitz  # PyMuPDF
from werkzeug.utils import secure_filename
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

import nltk
nltk.download('punkt') 


import threading
import subprocess


def download_spacy_model():
    try:
        print("Checking if spaCy model exists...")
        subprocess.run(["python", "-m", "spacy", "validate"], check=True)
    except Exception:
        print("Downloading spaCy model...")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)

# Start a background thread to download the model
threading.Thread(target=download_spacy_model, daemon=True).start()

UPLOAD_FOLDER = "/tmp/uploads"
OUTPUT_FOLDER = "/tmp/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved model
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
model.load_state_dict(torch.load('roberta_15epoch_model.pth', map_location=device))
model.to(device)
model.eval()

# Load Sentence Transformer for contextual embeddings
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load spaCy for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

# Gender-related keywords
GENDER_KEYWORDS = {
    # Pronouns
    "he", "she", "him", "her", "his", "hers", 
    "himself", "herself", "ze", "zir", "xe", "xem", "hir", "hirs",

    # Gendered Nouns
    "man", "woman", "men", "women", "boy", "boys", "girl", "girls", "gentleman", "lady", 
    "ladies", "gentlemen", "guy", "gal", "dude", "chick", "lad", "lass",

    # Family Roles
    "father", "mother", "dad", "mom", "papa", "mama", 
    "fathers", "mothers", "daughters", "sons",
    "son", "daughter", "brother", "sister", "sibling", 
    "husband", "wife", "spouse", "partner",
    "aunt", "uncle", "nephew", "niece", "cousin",
    "grandfather", "grandmother", "grandson", "granddaughter", 
    "grandparent", "stepfather", "stepmother", "stepson", "stepdaughter",

    # Romantic & Relationship Terms
    "boyfriend", "girlfriend", "fiancé", "fiancée", 
    "bride", "groom", "lover", "sweetheart", "crush",

    # Gendered Titles & Formal Address
    "sir", "madam", "mr", "mrs", "miss", "ms", "ma'am", "lord", "lady", 
    "queen", "king", "prince", "princess", "duke", "duchess",

    # Gender & Identity Terms
    "masculine", "feminine", "androgynous", "nonbinary", "transgender", 
    "cisgender", "bigender", "genderfluid", "genderqueer", "agender",
    "two-spirit", "demiboy", "demigirl", "femme", "butch",

    # Societal & Cultural Terms
    "patriarchy", "matriarchy", "sexism", "feminism", "misogyny", "misandry",
    "mankind", "womankind", "humanity", "motherland", "fatherland",
    "alpha male", "alpha female", "tomboy", "girly", "manly",
    
    # Parenting & Childcare
    "maternal", "paternal", "motherhood", "fatherhood", "nurturing",
    
    # Miscellaneous
    "bachelor", "spinster", "widow", "widower", "divorcé", "divorcée"
}

def detect_bias(text):
    """Detect bias in a given text using the trained model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)[0].tolist()
    return {"neutral": scores[0], "biased": scores[1]}

def counterfactual_test(text, bias_threshold=0.2):
    """Perform counterfactual testing by swapping gender terms."""
    swaps = {"he": "she", "she": "he", "his": "her", "her": "his", "man": "woman", 
             "woman": "man", "men": "women", "women": "men", "boy": "girl", "girl": "boy",
             "guys": "gals", "gals": "guys", "gentleman": "lady", "lady": "gentleman",
             "father": "mother", "mother": "father", "son": "daughter", "daughter": "son",
             "husband": "wife", "wife": "husband", "brother": "sister", "sister": "brother"}
    
    words = text.split()
    flipped_text = " ".join([swaps.get(word.lower(), word) for word in words])
    
    orig_score = detect_bias(text)
    flipped_score = detect_bias(flipped_text)
    bias_difference = abs(orig_score["biased"] - flipped_score["biased"])
    
    is_biased = bias_difference >= bias_threshold
    
    return {
        "original": text,
        "flipped": flipped_text,
        "orig_bias_score": orig_score["biased"],
        "flipped_bias_score": flipped_score["biased"],
        "bias_difference": bias_difference,
        "is_biased": is_biased
    }

def is_gender_related(sentence):
    """Check if a sentence contains gender-related terms."""
    words = set(re.findall(r'\b\w+\b', sentence.lower()))
    return any(word in GENDER_KEYWORDS for word in words)

def detect_gender_bias_counterfactual(paragraph, bias_difference_threshold=0.2):
    """Detect gender bias in a paragraph using counterfactual testing."""
    sentences = sent_tokenize(paragraph)
    biased_sentences = []

    for sentence in sentences:
        if is_gender_related(sentence):
            result = counterfactual_test(sentence, bias_difference_threshold)
            if result["is_biased"]:
                biased_sentences.append(result["original"])

    return biased_sentences

def analyze_text_corpus_counterfactual(corpus_file, bias_difference_threshold=0.2):
    """Analyze a corpus of text for gender bias using counterfactual testing."""
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus = f.read()
        
        paragraphs = corpus.split('\n\n')
        all_biased_sentences = []
        
        for paragraph in paragraphs:
            if paragraph.strip():
                biased_sentences = detect_gender_bias_counterfactual(paragraph, bias_difference_threshold)
                all_biased_sentences.extend(biased_sentences)
        
        return all_biased_sentences
    except Exception as e:
        print(f"Error analyzing corpus: {str(e)}")
        return []

def detect_gender_bias(text, threshold=0.2):
    """Detect gender bias in text using the simple method."""
    sentences = sent_tokenize(text)
    return [s for s in sentences if detect_bias(s)["biased"] >= threshold]

def process_pdf(input_pdf, output_pdf, use_counterfactual=True):
    """Process PDF and highlight biased text."""
    doc = fitz.open(input_pdf)
    for page in doc:
        text = page.get_text("text")
        
        if use_counterfactual:
            biased_sentences = detect_gender_bias_counterfactual(text)
        else:
            biased_sentences = detect_gender_bias(text)
            
        for sentence in biased_sentences:
            text_instances = page.search_for(sentence)
            for inst in text_instances:
                page.add_highlight_annot(inst)
    
    doc.save(output_pdf)
    doc.close()

def process_text_file(input_file, output_file, use_counterfactual=True):
    """Process text file and write biased sentences to output file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if use_counterfactual:
            biased_sentences = detect_gender_bias_counterfactual(text)
        else:
            biased_sentences = detect_gender_bias(text)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Detected Gender Bias:\n\n")
            for i, sentence in enumerate(biased_sentences, 1):
                f.write(f"{i}. {sentence}\n\n")
        
        return biased_sentences
    except Exception as e:
        print(f"Error processing text file: {str(e)}")
        return []

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file part", 400
    
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"processed_{filename}")
    
    file.save(input_path)
    
    # Get method parameter (counterfactual or simple)
    use_counterfactual = request.form.get('method', 'counterfactual') == 'counterfactual'
    
    # Process based on file type
    if filename.lower().endswith('.pdf'):
        process_pdf(input_path, output_path, use_counterfactual)
    elif filename.lower().endswith(('.txt', '.md', '.csv')):
        output_path = os.path.join(OUTPUT_FOLDER, f"processed_{os.path.splitext(filename)[0]}.txt")
        process_text_file(input_path, output_path, use_counterfactual)
    else:
        return "Unsupported file format. Please upload PDF or text files.", 400
    
    return send_file(output_path, as_attachment=True)

@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    text = request.form.get('text', '')
    if not text:
        return "No text provided", 400
    
    use_counterfactual = request.form.get('method', 'counterfactual') == 'counterfactual'
    
    if use_counterfactual:
        biased_sentences = detect_gender_bias_counterfactual(text)
    else:
        biased_sentences = detect_gender_bias(text)
    
    return render_template("results.html", biased_sentences=biased_sentences)

@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
