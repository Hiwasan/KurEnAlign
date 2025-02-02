import argparse
import re
import sentencepiece as spm
import simalign
import os
import json
import sys 
from tqdm import tqdm
from docx import Document
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def extract_table_data(docx_path):
    """
    Extracts table data from a .docx file.

    Args:
        docx_path (str): Path to the .docx file.

    Returns:
        tuple: Two lists containing Mukri (source) and English (target) texts.
    """
    logging.info(f"Attempting to open document: {docx_path}")

    # Check if file exists
    if not os.path.exists(docx_path):
        logging.error(f"File does not exist: {docx_path}")
        return [], []

    # Check if file is readable
    if not os.access(docx_path, os.R_OK):
        logging.error(f"File is not readable: {docx_path}")
        return [], []

    try:
        doc = Document(docx_path)
        logging.info("Document opened successfully!")
    except Exception as e:
        logging.error(f"Error opening document: {e}")
        return [], []

    source_text = []
    target_text = []

    for table in doc.tables:
        logging.info(f"Processing table with {len(table.rows)} rows")
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            logging.info(f"Row cells: {cells}")
            if len(cells) == 4:  # Assuming the table has 4 columns: ID, Mukri, English, Timestamp
                source_text.append(cells[1])
                target_text.append(cells[2])

    return source_text, target_text

# Example usage
if __name__ == "__main__":
    docx_path = r"C:\NLP\Mukri\Mukri.docx"  # Replace with your file path
    source_text, target_text = extract_table_data(docx_path)

    if source_text and target_text:
        # Print extracted texts
        logging.info("Source (Mukri) Text:")
        for text in source_text:
            logging.info(text)
        
        logging.info("\nTarget (English) Text:")
        for text in target_text:
            logging.info(text)
    else:
        logging.info("No data extracted from the document.")

# Choose layer for embedding extraction
LAYER = 8

# Argument parsing
def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Align source and target texts using SimAlign.")
    parser.add_argument('-m', '--margin_file', type=str, required=True, help='Margin score file')
    parser.add_argument('-s', '--source_file', type=str, required=True, help='Source language file')
    parser.add_argument('-t', '--target_file', type=str, required=True, help='Target language file')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Output file path')
    parser.add_argument('--source_sp_model', type=str, default='source_sp.model', help='Path to source SentencePiece model')
    parser.add_argument('--target_sp_model', type=str, default='target_sp.model', help='Path to target SentencePiece model')
    parser.add_argument('--model', type=str, default='xlm-roberta-base', help='Model name for SimAlign')
    parser.add_argument('--layer', type=int, default=8, help='Layer index for embedding extraction')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for alignment computation')

    args = parser.parse_args()
    logging.info(f"Arguments parsed: {args}")  # 🔥 Debugging line
    return args

def simple_preprocess(sentence):
    """
    Simple preprocessing for sentences: remove punctuation.
    
    Args:
        sentence (str): Input sentence.
    
    Returns:
        str: Preprocessed sentence.
    """
    sentence = re.sub(r'[.,";?!]', '', sentence)
    return re.sub(r'\s+', ' ', sentence).strip()

def text_to_dict(text):
    """
    Convert the corpus into a dictionary.
    
    Args:
        text (str): The input text containing sentence pairs.
    
    Returns:
        dict: A dictionary mapping keys to sentences.
    """
    split_text = text.split('\n')[:-1]
    text_dict = {}
    for line in tqdm(split_text, desc="Processing text"):
        split_line = line.split('\t')
        if len(split_line) != 2:
            logging.warning(f"Line does not have exactly two elements: {line}")
            continue
        key, value = split_line
        text_dict[key] = value
    return text_dict

def read_file(file_path):
    """
    Read the content of a file.
    
    Args:
        file_path (str): Path to the file.
    
    Returns:
        str: File content.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_file(file_path, content):
    """
    Write content to a file.
    
    Args:
        file_path (str): Path to the file.
        content (str): Content to write.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
    except Exception as e:
        logging.error(f"Error writing to file {file_path}: {e}")
        raise

def tokenize_text(text, sp_model):
    """
    Tokenize text using SentencePiece.
    
    Args:
        text (str): Input text.
        sp_model (str): Path to SentencePiece model.
    
    Returns:
        list: List of tokens.
    """
    try:
        sp = spm.SentencePieceProcessor(model_file=sp_model)
        return sp.encode(text, out_type=str)
    except Exception as e:
        logging.error(f"Error tokenizing text with model {sp_model}: {e}")
        raise

def align_texts(source_tokens, target_tokens, model="xlm-roberta-base", matching_methods="mai"):
    """
    Align source and target tokens using SimAlign.
    
    Args:
        source_tokens (list): Source tokens.
        target_tokens (list): Target tokens.
        model (str): Model name for SimAlign.
        matching_methods (str): Matching methods for alignment.
    
    Returns:
        dict: Alignment results.
    """
    aligner = simalign.SentenceAligner(model=model, token_type="bpe", matching_methods=matching_methods)
    return aligner.get_word_aligns(" ".join(source_tokens), " ".join(target_tokens))

def align_texts_batched(source_dict, target_dict, args):
    """
    Align texts in batches for performance optimization.
    
    Args:
        source_dict (dict): Source sentences.
        target_dict (dict): Target sentences.
        args (argparse.Namespace): Command-line arguments.
    
    Returns:
        dict: Dictionary of alignments.
    """
    alignments = {}
    batch_size = args.batch_size
    keys = list(source_dict.keys())
    for i in tqdm(range(0, len(keys), batch_size), desc="Aligning texts"):
        batch_keys = keys[i:i + batch_size]
        for key in batch_keys:
            source_tokens = source_dict[key]
            target_tokens = target_dict[key]
            try:
                alignments[key] = align_texts(source_tokens, target_tokens, model=args.model)
            except Exception as e:
                logging.error(f"Error aligning text pair {key}: {e}")
                alignments[key] = None
    return alignments

def clean_text(text):
    """Clean text by removing non-UTF-8 characters and normalizing whitespace."""
    # Remove or replace problematic characters
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    # Normalize whitespace
    text = ' '.join(text.split())
    return text

def train_sentencepiece_models(source_text, target_text, base_path):
    """Train SentencePiece models with proper text handling."""
    import sentencepiece as spm
    
    # Create temporary files for training
    source_train_file = os.path.join(base_path, 'source_train.txt')
    target_train_file = os.path.join(base_path, 'target_train.txt')
    
    try:
        # Write cleaned text to temporary files
        with open(source_train_file, 'w', encoding='utf-8') as f:
            for line in source_text.split('\n'):
                if line.strip():
                    cleaned_line = clean_text(line)
                    f.write(cleaned_line + '\n')
        
        with open(target_train_file, 'w', encoding='utf-8') as f:
            for line in target_text.split('\n'):
                if line.strip():
                    cleaned_line = clean_text(line)
                    f.write(cleaned_line + '\n')
        
        # Train source model
        source_model_path = os.path.join(base_path, 'source_sp')
        spm.SentencePieceTrainer.train(
            input=source_train_file,
            model_prefix=source_model_path,
            vocab_size=50,  # Increased vocabulary size
            character_coverage=0.9995,
            model_type='unigram',
            input_format='text',
            pad_id=3,
            unk_id=0,
            bos_id=1,
            eos_id=2
        )
        
        # Train target model
        target_model_path = os.path.join(base_path, 'target_sp')
        spm.SentencePieceTrainer.train(
            input=target_train_file,
            model_prefix=target_model_path,
            vocab_size=50,  # Increased vocabulary size
            character_coverage=0.9995,
            model_type='unigram',
            input_format='text',
            pad_id=3,
            unk_id=0,
            bos_id=1,
            eos_id=2
        )
        
        return f"{source_model_path}.model", f"{target_model_path}.model"
        
    except Exception as e:
        logging.error(f"Error during SentencePiece training: {e}")
        raise
    finally:
        # Clean up temporary files
        for file in [source_train_file, target_train_file]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except Exception as e:
                    logging.warning(f"Could not remove temporary file {file}: {e}")

def print_alignments_with_words(source_text, target_text, alignments):
    source_sentences = source_text.split('\n')
    target_sentences = target_text.split('\n')

    for key, alignment in alignments.items():
        if alignment is not None:
            src_sentence = source_sentences[int(key.split('_')[1]) - 1].split()
            tgt_sentence = target_sentences[int(key.split('_')[1]) - 1].split()
            print(f"Alignment for sentence pair {key}:")
            print(f"  Source sentence: {' '.join(src_sentence)}")
            print(f"  Target sentence: {' '.join(tgt_sentence)}")
            print(f"  Source sentence length: {len(src_sentence)}")
            print(f"  Target sentence length: {len(tgt_sentence)}")
            for method, pairs in alignment.items():
                print(f"  Method: {method}")
                for src_idx, tgt_idx in pairs:
                    if src_idx < len(src_sentence) and tgt_idx < len(tgt_sentence):
                        print(f"    {src_sentence[src_idx]} -> {tgt_sentence[tgt_idx]}")
                    else:
                        print(f"    Index out of range: src_idx={src_idx}, tgt_idx={tgt_idx}")
            print()

def main():
    args = parse_args()

    logging.info("Starting alignment process...")

    # Extract data from the .docx file
    docx_path = r"C:\NLP\Mukri\Mukri.docx"
    source_text, target_text = extract_table_data(docx_path)

    if not source_text or not target_text:
        logging.error("Failed to extract data from the document.")
        return

    logging.info(f"Number of extracted source sentences: {len(source_text)}")
    logging.info(f"Number of extracted target sentences: {len(target_text)}")

    # Clean and format the texts
    source_lines = []
    target_lines = []
    
    for i, (src, tgt) in enumerate(zip(source_text, target_text)):
        if src.strip() and tgt.strip():
            source_lines.append(f"line_{i+1}\t{src.strip()}")
            target_lines.append(f"line_{i+1}\t{tgt.strip()}")
    
    source_text_formatted = '\n'.join(source_lines)
    target_text_formatted = '\n'.join(target_lines)

    logging.info("Writing formatted texts to files...")
    
    # Write formatted texts to files
    write_file(args.source_file, source_text_formatted)
    write_file(args.target_file, target_text_formatted)
    write_file(args.margin_file, '')  # Empty margin file for now

    # Create dictionaries
    source_dict = text_to_dict(source_text_formatted)
    target_dict = text_to_dict(target_text_formatted)

    logging.info(f"Number of sentence pairs for alignment: {len(source_dict)}")

    # Train SentencePiece models
    base_path = os.path.dirname(args.source_file)
    try:
        source_model, target_model = train_sentencepiece_models(
            '\n'.join(source_dict.values()),
            '\n'.join(target_dict.values()),
            base_path
        )
    except Exception as e:
        logging.error(f"Failed to train SentencePiece models: {e}")
        return

    # Update model paths
    args.source_sp_model = source_model
    args.target_sp_model = target_model

    # Load SentencePiece models
    source_sp = spm.SentencePieceProcessor(model_file=args.source_sp_model)
    target_sp = spm.SentencePieceProcessor(model_file=args.target_sp_model)

    # Tokenize texts
    source_tokens = {k: tokenize_text(v, args.source_sp_model) for k, v in source_dict.items()}
    target_tokens = {k: tokenize_text(v, args.target_sp_model) for k, v in target_dict.items()}

    # Initialize SimAlign
    aligner = simalign.SentenceAligner(model=args.model, token_type="bpe", matching_methods="mai")

    # Align texts in batches
    alignments = {}
    keys = list(source_tokens.keys())
    batch_size = args.batch_size
    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i:i + batch_size]
        for key in batch_keys:
            try:
                src_sentence = simple_preprocess(source_dict[key])
                tgt_sentence = simple_preprocess(target_dict[key])
                src_tokens = source_sp.encode(src_sentence, out_type=str)
                tgt_tokens = target_sp.encode(tgt_sentence, out_type=str)
                alignments[key] = aligner.get_word_aligns(" ".join(src_tokens), " ".join(tgt_tokens))
            except Exception as e:
                logging.error(f"Error aligning text pair {key}: {e}")
                alignments[key] = None

    # Write output
    output_file_path = args.output_file
    write_file(output_file_path, json.dumps(alignments, ensure_ascii=False, indent=4))
    logging.info(f"Alignment results saved to {output_file_path}")

    # Log summary
    error_count = sum(1 for v in alignments.values() if v is None)
    logging.info(f"Alignment completed. Total pairs: {len(alignments)}, Failed pairs: {error_count}")

    # Print alignments with words
    print_alignments_with_words(source_text_formatted, target_text_formatted, alignments)

if __name__ == "__main__":
    import sys
    base_path = r"C:\Users\hivad\OneDrive\Documents\DFG bilateral project\KurEnAlign\data"
    os.makedirs(base_path, exist_ok=True)
    
    # Create initial files if they don't exist
    def create_initial_files(base_path):
        files = ['margin_file.txt', 'source_file.txt', 'target_file.txt']
        for file in files:
            file_path = os.path.join(base_path, file)
            if not os.path.exists(file_path):
                logging.info(f"Creating empty file: {file_path}")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('')
    
    create_initial_files(base_path)
    
    sys.argv = [
        'sentence_aligner.py',
        '-m', os.path.join(base_path, 'margin_file.txt'),
        '-s', os.path.join(base_path, 'source_file.txt'),
        '-t', os.path.join(base_path, 'target_file.txt'),
        '-o', os.path.join(base_path, 'output_file.txt')
    ]
    
    main()