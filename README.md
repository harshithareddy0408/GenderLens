# GenderLens

**GenderLens** is a web-based application that detects gender bias in text and PDF documents. It leverages a fine-tuned RoBERTa model to classify biased statements and highlight gendered content, with a special focus on educational materials like Indian textbooks. The system was developed as part of a research initiative exploring gender and queer bias in NLP.

---

## 💡 Features

- 🔍 **Bias Detection in Text** – Analyze input text for gender bias.
- 📄 **PDF Upload** – Upload PDF documents and detect biased sentences.
- ✨ **Highlighted Output** – Download processed PDFs with biased content marked.
- 🤖 **NLP-Powered** – Uses a fine-tuned RoBERTa model for bias classification.
- 🌐 **Simple Web Interface** – User-friendly frontend with HTML, CSS, and JavaScript.

---

## 📁 Project Structure

| File                             | Description |
|----------------------------------|-------------|
| `app.py`                         | Flask backend and model logic |
| `index.html`                     | Main interface for bias detection |
| `about.html`                     | About the project |
| `features.html`                  | Project features and capabilities |
| `contact.html`                   | Contact or feedback form |
| `script.js`                      | Frontend logic for form handling |
| `styles.css`                     | Styling for the web pages |
| `gender_bias_example.pdf`        | Sample input PDF |
| `processed_gender_bias_example.pdf` | Output PDF with highlighted bias |
| `requirements.txt`               | Python dependencies |
| `README.md`                      | Project documentation |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/genderlens.git
cd genderlens
