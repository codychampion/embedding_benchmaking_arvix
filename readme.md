# Academic Embedding Model Evaluator

A comprehensive tool for evaluating embedding models on academic paper similarity tasks. This project helps assess how well different embedding models capture semantic relationships between academic paper titles and abstracts across various scientific fields.

## Features

- Automated paper retrieval from arXiv across multiple scientific domains
- Support for multiple embedding models (including specialized scientific models)
- Efficient embedding generation with caching
- Comprehensive evaluation metrics:
  - Title-to-abstract similarity within papers
  - Cross-paper similarity analysis
  - Same-field vs. cross-field comparisons
- Results visualization and model leaderboard generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/academic-embedding-evaluator.git
cd academic-embedding-evaluator
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your HuggingFace token (get it from https://huggingface.co/settings/tokens)

## Usage

The tool provides a command-line interface with several commands:

### Basic Evaluation

Run the evaluation with default settings:
```bash
python embedding_evaluator.py evaluate
```

### Custom Configuration

1. Generate a template configuration file:
```bash
python embedding_evaluator.py init-config config.yaml
```

2. Edit the configuration file to customize models and fields

3. Run evaluation with custom config:
```bash
python embedding_evaluator.py evaluate --config config.yaml
```

### Additional Options

```bash
# Show help message
python embedding_evaluator.py --help

# Show help for evaluate command
python embedding_evaluator.py evaluate --help

# Customize cache and output directories
python embedding_evaluator.py evaluate \
    --cache-dir ./cache \
    --output-dir ./results \
    --max-papers 50 \
    --config config.yaml
```
```

## Output

The evaluator generates two main output files:

1. `embedding_comparison_results.csv`: Detailed metrics for each model
2. `model_leaderboard.csv`: Simplified leaderboard with key performance indicators

### Evaluation Metrics

- **Title-Own Abstract**: Similarity between a paper's title and its abstract
- **Title-Different Abstract**: Similarity between a paper's title and other papers' abstracts
- **Abstract-Abstract**: Similarity between abstracts of different papers
- **Field Comparisons**: Similarities within and across scientific fields

## Supported Models

The evaluator includes various types of embedding models:

### Scientific/Academic Specialized
- allenai/specter
- allenai/scibert_scivocab_uncased
- stanford/scifive-base

### General Purpose
- sentence-transformers/all-mpnet-base-v2
- sentence-transformers/all-MiniLM-L6-v2

### E5 Family
- intfloat/e5-base
- intfloat/e5-large

### BGE Models
- BAAI/bge-base-en
- BAAI/bge-large-en

## Scientific Fields

The evaluation covers various scientific domains including:
- Computer & Information Science
- Engineering
- Physical Sciences
- Biological Sciences
- Interdisciplinary Fields

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{academic_embedding_evaluator,
  author = {Your Name},
  title = {Academic Embedding Model Evaluator},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/academic-embedding-evaluator}
}
```

## Acknowledgments

- ArXiv for providing access to academic papers
- HuggingFace for hosting the embedding models
- All model creators and contributors

## Note

This tool is designed for research purposes and evaluates publicly available models. Performance may vary depending on the specific use case and domain.