import argparse
import logging
import os
import re
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from docx import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class EmbeddingLoader:
    """
    Load and manage embeddings from pre-trained models.
    """
    def __init__(self, model: str, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), layer: int=8):
        self.model = model
        self.device = device
        self.layer = layer
        config = AutoConfig.from_pretrained(model, output_hidden_states=True)
        self.emb_model = AutoModel.from_pretrained(model, config=config)
        self.emb_model.eval()
        self.emb_model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        logging.info(f"Initialized EmbeddingLoader with model: {self.model}")

    def get_embed_list(self, sent_batch):
        """
        Get embeddings for a batch of sentences.
        """
        try:
            with torch.no_grad():
                inputs = self.tokenizer(sent_batch, padding=True, truncation=True, return_tensors="pt")
                hidden = self.emb_model(**inputs.to(self.device))["hidden_states"]
                if self.layer >= len(hidden):
                    raise ValueError(f"Specified layer {self.layer} exceeds model's {len(hidden)} layers.")
                return hidden[self.layer][:, 1:-1, :]  # Exclude [CLS] and [SEP] tokens
        except Exception as e:
            logging.error(f"Error computing embeddings: {e}")
            return None

def extract_table_data(docx_path):
    """
    Extracts table data from a .docx file.
    """
    logging.info(f"Attempting to open document: {docx_path}")

    if not os.path.exists(docx_path):
        logging.error(f"File does not exist: {docx_path}")
        return [], []

    if not os.access(docx_path, os.R_OK):
        logging.error(f"File is not readable: {docx_path}")
        return [], []

    try:
        doc = Document(docx_path)
        logging.info("Document opened successfully!")
    except Exception as e:
        logging.error(f"Error opening document: {e}")
        return [], []

    source_text, target_text = [], []
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if len(cells) == 4:
                source_text.append(cells[1])
                target_text.append(cells[2])

    return source_text, target_text

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Extract embeddings using XLM-Roberta.")
    parser.add_argument('-i', '--input_file', type=str, default=r"C:\NLP\Mukri\Mukri.docx", help='Input document file')
    parser.add_argument('-o', '--output_file', type=str, default="output.pt", help='Output file path')  # <-- DEFAULT ADDED
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base', help='Model name for XLM-Roberta')
    parser.add_argument('--layer', type=int, default=8, help='Layer index for embedding extraction')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding extraction')
    return parser.parse_args()

def get_embedding(sentence, xlmr_embeddings):
    """
    Compute the average embedding for a single sentence.
    """
    embedding_size = 768
    s_embedding = xlmr_embeddings.get_embed_list([sentence])
    if s_embedding is None or s_embedding.size(dim=0) != 1:
        logging.warning(f"Invalid embedding for sentence: {sentence}")
        return None
    np_embedding = s_embedding.cpu().detach().numpy()[0].mean(axis=0)
    return [f"{embed_value:.6f}" for embed_value in np_embedding] if len(np_embedding) == embedding_size else None

def save_embeddings(output_file, sentence_list, model_name, layer):
    """
    Save sentence embeddings in a batch manner.
    """
    xlmr_embeddings = EmbeddingLoader(model_name, layer=layer)
    all_embeddings = []
    for sentence in sentence_list:
        word_embeddings = extract_embeddings(sentence, model_name, layer)
        all_embeddings.append(word_embeddings)
    torch.save(all_embeddings, output_file)

def extract_embeddings(text, model_name, layer):
    """
    Extract embeddings from text using a pre-trained model.
    
    Args:
        text (str): Input text.
        model_name (str): Pre-trained model name.
        layer (int): Layer index for embedding extraction.
    
    Returns:
        list: List of tuples containing words and their embeddings.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    embeddings = hidden_states[layer]
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    word_embeddings = [(token, embedding) for token, embedding in zip(tokens, embeddings[0])]
    
    return word_embeddings

def main():
    args = parse_args()

    # Read input file
    try:
        if args.input_file.endswith('.docx'):
            source_text, target_text = extract_table_data(args.input_file)
            if not source_text or not target_text:
                logging.error("No valid data extracted from the document.")
                return
            sentence_list = source_text + target_text
        else:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                sentence_list = f.read().strip().split("\n")
    except Exception as e:
        logging.error(f"Error reading file {args.input_file}: {e}")
        return

    logging.info("Starting embedding extraction...")

    save_embeddings(args.output_file, sentence_list, args.model_name, args.layer)

    logging.info(f"Embeddings saved to {args.output_file}")

if __name__ == "__main__":
    main()