#!/bin/bash

# Set working directory
cd "$(dirname "$0")" || { echo "Failed to set working directory"; exit 1; }

# Install dependencies
pip install -r requirements.txt || { echo "Failed to install dependencies"; exit 1; }

# Source environment variables
if [[ -f ./environment.sh ]]; then
    source ./environment.sh
else
    echo "Environment file not found!"
    exit 1
fi

# Check for Python executable
PYTHON=${PYTHON:-python3}
command -v $PYTHON >/dev/null 2>&1 || { echo "Python executable not found!"; exit 1; }

# Parse command-line arguments
while getopts ":s:t:d:e:o:p:" opt; do
    case ${opt} in
        s ) SRC_LANG=$OPTARG ;;
        t ) TRG_LANG=$OPTARG ;;
        d ) DATA_DIR=$OPTARG ;;
        e ) EMBEDDINGS_DIR=$OPTARG ;;
        o ) OUTPUT_DIR=$OPTARG ;;
        p ) PREFIX=$OPTARG ;;
        \? ) echo "Usage: $0 [-s source_lang] [-t target_lang] [-d data_dir] [-e embeddings_dir] [-o output_dir] [-p prefix]"; exit 1 ;;
    esac
done

# Validate required arguments
[[ -z "${SRC_LANG}" || -z "${TRG_LANG}" || -z "${DATA_DIR}" || -z "${EMBEDDINGS_DIR}" || -z "${OUTPUT_DIR}" || -z "${PREFIX}" ]] && \
    { echo "Missing required arguments!"; exit 1; }

# Create directories
mkdir -p "${EMBEDDINGS_DIR}/${SRC_LANG}-${TRG_LANG}"
mkdir -p "${OUTPUT_DIR}"

# Check if input files exist
if [[ ! -f "${DATA_DIR}/${SRC_LANG}-${TRG_LANG}.${SRC_LANG}" ]]; then
    echo "Source language file not found!"
    exit 1
fi

if [[ ! -f "${DATA_DIR}/${SRC_LANG}-${TRG_LANG}.${TRG_LANG}" ]]; then
    echo "Target language file not found!"
    exit 1
fi

# Generate document embeddings
echo "Generating document embeddings..."
$PYTHON contextual_document_embeddings.py \
    --input_file "${DATA_DIR}/${SRC_LANG}-${TRG_LANG}.${SRC_LANG}" \
    --output_file "${EMBEDDINGS_DIR}/${SRC_LANG}-${TRG_LANG}/${PREFIX}${SRC_LANG}.vec" -m 'xlmr' || \
    { echo "Embedding generation failed for source language!"; exit 1; }

$PYTHON contextual_document_embeddings.py \
    --input_file "${DATA_DIR}/${SRC_LANG}-${TRG_LANG}.${TRG_LANG}" \
    --output_file "${EMBEDDINGS_DIR}/${SRC_LANG}-${TRG_LANG}/${PREFIX}${TRG_LANG}.vec" -m 'xlmr' || \
    { echo "Embedding generation failed for target language!"; exit 1; }

# Find nearest neighbors
echo "Finding nearest neighbors..."
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/bilingual_nearest_neighbor.py \
    --source_embeddings "${EMBEDDINGS_DIR}/${SRC_LANG}-${TRG_LANG}/${PREFIX}${SRC_LANG}.vec" \
    --target_embeddings "${EMBEDDINGS_DIR}/${SRC_LANG}-${TRG_LANG}/${PREFIX}${TRG_LANG}.vec" \
    --output "${OUTPUT_DIR}/DOC.${PREFIX}${SRC_LANG}-${TRG_LANG}.sim" --knn 10 -m csls --cslsknn 20 || \
    { echo "Nearest neighbor search failed!"; exit 1; }

# Filter and evaluate
echo "Filtering and evaluating alignments..."
$PYTHON scripts/filter.py \
    -i "${OUTPUT_DIR}/DOC.${PREFIX}${SRC_LANG}-${TRG_LANG}.sim" \
    -m threshold -th 0.7 \
    -o "${OUTPUT_DIR}/filtered.${PREFIX}${SRC_LANG}-${TRG_LANG}.sim" || \
    { echo "Filtering failed!"; exit 1; }

$PYTHON scripts/bucc_f-score.py \
    -p "${OUTPUT_DIR}/filtered.${PREFIX}${SRC_LANG}-${TRG_LANG}.sim" \
    -g "${DATA_DIR}/${SRC_LANG}-${TRG_LANG}.gold" > "${OUTPUT_DIR}/evaluation_results.txt" || \
    { echo "Evaluation failed!"; exit 1; }

echo "Pipeline completed successfully. Results saved to ${OUTPUT_DIR}."
