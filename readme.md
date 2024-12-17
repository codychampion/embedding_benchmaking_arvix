# Academic Embedding Model Evaluator

**Author**: Cody Champion

## Current Model Leaderboard

| Model  | Score | Own Title-Abstract | Same Field Separation | Avg Std |
|--------|--------|-------------------|---------------------|---------|
| MPNet  | 0.466  | 0.718            | 0.393              | 0.088   |
| MiniLM | 0.466  | 0.718            | 0.393              | 0.088   |
| E5-Base| 0.264  | 0.858            | 0.074              | 0.025   |

Metrics explained:
- **Score**: Overall performance score (higher is better)
- **Own Title-Abstract**: Similarity between a paper's title and its own abstract
- **Same Field Separation**: Difference in similarity between same-field and different-field papers
- **Avg Std**: Average standard deviation across all metrics (lower means more consistent)

---

A powerful command-line tool for evaluating and comparing embedding models on academic paper similarity tasks. This tool helps researchers and practitioners assess how well different embedding models perform at capturing semantic relationships between academic papers.

## Features

- 📊 Comprehensive model evaluation across multiple academic fields
- 🔄 Support for various embedding models (Hugging Face models supported)
- 📑 Multiple comparison metrics:
  - Title-to-Abstract similarity
  - Cross-paper Abstract comparisons
  - Same-field vs Different-field distinctions
- 💾 Efficient caching system for embeddings
- 📈 Detailed performance leaderboard

## Hardware Acceleration & Cloud Integration

### GPU Support (Optional)
The tool works efficiently on both CPU and GPU setups. While GPU acceleration can speed up embedding generation, it's completely optional. The tool automatically detects and uses the appropriate hardware:
```python
device = "cuda:0" if torch.cuda.is_available() else "cpu"
```

Key points about hardware:
- CPU-only setup works well for most use cases
- GPU acceleration available but not required
- Automatic hardware detection and optimization
- All models support both CPU and GPU execution

Requirements:
- Basic: Any modern CPU
- Optional GPU acceleration:
  - CUDA-compatible GPU
  - PyTorch with CUDA support (installed via requirements.txt)
  - Sufficient GPU memory (varies by model size)

### AWS Bedrock Integration
The tool includes AWS Bedrock's Titan Embeddings model, which is not just a baseline but often the most powerful model in the evaluation. This provides:
- State-of-the-art embedding quality
- Consistent high performance
- Cloud-based processing (no local compute requirements)
- Production-ready capabilities

To use AWS Bedrock:
1. Configure AWS credentials in your environment
2. Include 'Bedrock' in your model configuration:
```yaml
models:
  'Bedrock': 'Bedrock'  # AWS Titan Embeddings model
```

The Bedrock model offers several advantages:
- 512-dimensional embeddings optimized for semantic similarity
- Normalized vectors for consistent comparisons
- Industry-leading performance on academic text
- No local compute or GPU requirements
- Highly scalable cloud-based processing

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd embedding-benchmarking-arxiv
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env-example .env
```
Edit `.env` and add your HuggingFace token:
```
HUGGINGFACE_TOKEN=your_token_here
```

## Configuration

The tool uses a YAML configuration file to specify models and research fields for evaluation. Create your config file:

```bash
python embedding_evaluator.py init-config config.yaml
```

Example configuration:
```yaml
models:
  # Cloud Provider Models
  'Bedrock': 'Bedrock'  # Amazon's Titan for comparison
  
  # Scientific/Academic Specialized
  'Specter': 'allenai/specter'  # Specialized for academic paper similarity
  'SciBERT': 'allenai/scibert_scivocab_uncased'  # Scientific vocabulary
  'BioLinkBERT': 'michiyasunaga/BioLinkBERT-base'  # Biomedical literature
  'BiomedNLP': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'  # Medical abstracts
  'SciFive': 'stanford/scifive-base'  # Scientific papers
  
  # Strong General Purpose Models
  'MPNet': 'sentence-transformers/all-mpnet-base-v2'  # Strong on academic text
  'MiniLM': 'sentence-transformers/all-MiniLM-L6-v2'  # Efficient, good for abstracts
  'RoBERTa': 'sentence-transformers/all-roberta-large-v1'
  
  # E5 Family
  'E5-Base': 'intfloat/e5-base'  # Strong academic performance
  'E5-Large': 'intfloat/e5-large'  # Larger version
  
  # BGE Models
  'BGE-Base': 'BAAI/bge-base-en'  # Good for technical content
  'BGE-Large': 'BAAI/bge-large-en'  # Larger version

fields:
  # Computer & Information Science (CISE)
  - "artificial intelligence"
  - "machine learning"
  - "computer vision"
  - "natural language processing"
  - "cybersecurity"
  - "quantum computing"
  - "robotics"
  
  # Engineering (ENG)
  - "biomedical engineering"
  - "nanotechnology"
  - "materials science"
  - "semiconductor physics"
  - "renewable energy"
  
  # Mathematical & Physical Sciences (MPS)
  - "quantum physics"
  - "particle physics"
  - "condensed matter physics"
  - "materials chemistry"
  - "astronomical sciences"
  
  # Biological Sciences (BIO)
  - "molecular biology"
  - "neuroscience"
  - "genomics"
  - "synthetic biology"
  - "bioinformatics"
  
  # Social & Behavioral Sciences (SBE)
  - "computational social science"
  - "cognitive science"
  - "data science"
  
  # Geosciences (GEO)
  - "climate science"
  - "atmospheric science"
  - "environmental science"
  
  # Emerging Technologies
  - "quantum information science"
  - "advanced manufacturing"
  - "biotechnology"
  
  # Cross-Cutting & Interdisciplinary
  - "computational biology"
  - "quantum materials"
  - "sustainable energy"
  - "artificial intelligence ethics"
```

## Usage

### Basic Usage

Run the evaluation with default settings:
```bash
python embedding_evaluator.py evaluate
```

### Advanced Options

```bash
python embedding_evaluator.py evaluate \
  --cache-dir embedding_cache \
  --max-papers 100 \
  --config custom_config.yaml \
  --output-dir results
```

Parameters:
- `--cache-dir`: Directory for storing embedding cache
- `--max-papers`: Maximum number of papers per field
- `--config`: Path to YAML configuration file
- `--output-dir`: Directory for saving results

## Example Output

When you run the evaluator, you'll see progress updates and results like this:

```
🚀 Starting Academic Embedding Model Evaluation
==================================================
📚 Loaded configuration from config.yaml
🤖 Models to evaluate: 3
🔬 Research fields: 3

📊 Total papers collected: 9
  Fetching papers for: biology              ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
  Processing paper 9/9 for intfloat/e5-base ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%


╭───────────────────────────────────────────────── 📊 Results ─────────────────────────────────────────────────╮
│                                            Model Comparison Results                                            │
│ ┏━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓ │
│ ┃ Model   ┃ Title-… ┃ Title-… ┃ Title-… ┃ Title-… ┃ Title-… ┃ Title-… ┃ Abstra… ┃ Abstra… ┃ Abstra… ┃ Abstr… ┃ │
│ ┃         ┃ Abstra… ┃ Abstra… ┃ (Same   ┃ (Same   ┃ (Diff   ┃ (Diff   ┃ (Same   ┃ (Same   ┃ (Diff   ┃ (Diff  ┃ │
│ ┃         ┃ Mean    ┃ Std     ┃ Mean    ┃ Std     ┃ Mean    ┃ Std     ┃ Mean    ┃ Std     ┃ Mean    ┃ Std    ┃ │
│ ┡━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩ │
│ │ MPNet   │ 0.718   │ 0.084   │ 0.361   │ 0.059   │ 0.036   │ 0.087   │ 0.465   │ 0.087   │ 0.072   │ 0.125  │ │
│ │ MiniLM  │ 0.718   │ 0.084   │ 0.361   │ 0.059   │ 0.036   │ 0.087   │ 0.465   │ 0.087   │ 0.072   │ 0.125  │ │
│ │ E5-Base │ 0.858   │ 0.029   │ 0.772   │ 0.022   │ 0.710   │ 0.031   │ 0.846   │ 0.018   │ 0.772   │ 0.024  │ │
│ └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴────────┘ │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

💾 Results saved to: embedding_comparison_results.csv


╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                           🏆 Model Leaderboard                                                  │
│ ┏━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓                                     │
│ ┃ Model   ┃ Score ┃ Own Title-Abstract ┃ Same Field Separation ┃ Avg Std ┃                                     │
│ ┡━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩                                     │
│ │ MPNet   │ 0.466 │ 0.718              │ 0.393                 │ 0.088   │                                     │
│ │ MiniLM  │ 0.466 │ 0.718              │ 0.393                 │ 0.088   │                                     │
│ │ E5-Base │ 0.264 │ 0.858              │ 0.074                 │ 0.025   │                                     │
│ └─────────┴───────┴────────────────────┴───────────────────────┴─────────┘                                     │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

💾 Leaderboard saved to: model_leaderboard.csv

✨ Evaluation complete!
==================================================
```

## Output & Metrics

The evaluator generates two main output files:

1. `embedding_comparison_results.csv`: Detailed metrics including:
   - Title-Own Abstract similarity
   - Title-Different Abstract similarity (same/different fields)
   - Abstract-Abstract similarity (same/different fields)
   - Standard deviations for all metrics

2. `model_leaderboard.csv`: Aggregated performance scores considering:
   - Title-Abstract matching accuracy
   - Field distinction capability
   - Consistency (via standard deviations)
   - Overall separation between related/unrelated papers

## License

This project is licensed under the terms of the license included in the repository. See [license.md](license.md) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{embedding_benchmarking_arxiv,
  title = {Academic Embedding Model Evaluator},
  author = {Champion, Cody},
  year = {2024},
  description = {A tool for evaluating embedding models on academic paper similarity tasks}
}
