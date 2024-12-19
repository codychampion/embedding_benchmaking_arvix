# 🏆 Academic Embedding Model Evaluator

**Author**: Cody Champion

## Current Model Leaderboard

Our sophisticated leaderboard system shows exactly how different AI models perform at understanding research papers:

```
╭────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                    🏆 Model Leaderboard                                         │
│ ┏━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓                  │
│ ┃ Model   ┃ Score ┃ Own Title-Abstract ┃ Same Field Separation ┃ Avg Std ┃                  │
│ ┡━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩                  │
│ │ MPNet   │ 0.466 │ 0.718              │ 0.393                 │ 0.088   │                  │
│ │ MiniLM  │ 0.466 │ 0.718              │ 0.393                 │ 0.088   │                  │
│ │ E5-Base │ 0.264 │ 0.858              │ 0.074                 │ 0.025   │                  │
│ └─────────┴───────┴────────────────────┴───────────────────────┴─────────┘                  │
╰────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Understanding the Metrics:
- **Score**: Overall performance score (higher is better)
- **Own Title-Abstract**: How well the model connects a paper's title with its content
- **Same Field Separation**: How accurately it distinguishes between related and unrelated papers
- **Avg Std**: Consistency of the model (lower means more reliable)

## 🎯 Beautiful Real-Time Progress Tracking

Watch the analysis happen in real-time with our elegant console interface:

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
│                                            Model Comparison Results                                           │
│ ┏━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓    │
│ ┃ Model   ┃ Title-… ┃ Title-… ┃ Title-… ┃ Title-… ┃ Title-… ┃ Title-… ┃ Abstra… ┃ Abstra… ┃ Abstra… ┃    │
│ ┡━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩    │
│ │ MPNet   │ 0.718   │ 0.084   │ 0.361   │ 0.059   │ 0.036   │ 0.087   │ 0.465   │ 0.087   │ 0.072   │    │
│ │ MiniLM  │ 0.718   │ 0.084   │ 0.361   │ 0.059   │ 0.036   │ 0.087   │ 0.465   │ 0.087   │ 0.072   │    │
│ │ E5-Base │ 0.858   │ 0.029   │ 0.772   │ 0.022   │ 0.710   │ 0.031   │ 0.846   │ 0.018   │ 0.772   │    │
│ └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## 📊 Professional Experiment Tracking

Every experiment is automatically documented with detailed statistics and configurations:

### 📈 Collection Statistics
```yaml
# collection_statistics.yaml
date_range:
  earliest: '2024-12-06'
  latest: '2024-12-18'
papers_per_field:
  artificial intelligence: 25
  biology: 25
  machine learning: 25
token_statistics:
  max: 407
  mean: 260.76
  median: 254.0
  min: 81
  std: 67.19
total_papers: 75
```

## 🚀 Key Features

- 📊 Comprehensive model evaluation across multiple academic fields
- 🔄 Support for various embedding models (Hugging Face models supported)
- 📑 Multiple comparison metrics
- 💾 Efficient caching system
- 📈 Detailed performance leaderboard
- 🎯 Beautiful progress tracking
- 📊 Professional experiment logging

## 🔧 Hardware Acceleration & Cloud Integration

### 🖥️ GPU Support (Optional)
Works efficiently on both CPU and GPU:
- Automatic hardware detection
- CPU-only setup works well for most cases
- Optional GPU acceleration for faster processing
- All models support both CPU and GPU execution

Requirements:
- Basic: Any modern CPU
- Optional GPU: CUDA-compatible GPU with PyTorch support

### ☁️ AWS Bedrock Integration
Includes AWS Bedrock's Titan Embeddings model:
- State-of-the-art embedding quality
- Cloud-based processing
- No local compute requirements
- Production-ready capabilities

## 🛠️ Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd embedding-benchmarking-arxiv
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment:
```bash
cp .env-example .env
# Add your HuggingFace token to .env
```

## ⚙️ Configuration

Create your config file with our comprehensive model selection:

```yaml
models:
  # Cloud Provider Models
  'Bedrock': 'Bedrock'  # Amazon's Titan
  
  # Scientific/Academic Specialized
  'Specter': 'allenai/specter'
  'SciBERT': 'allenai/scibert_scivocab_uncased'
  'BioLinkBERT': 'michiyasunaga/BioLinkBERT-base'
  
  # Strong General Purpose Models
  'MPNet': 'sentence-transformers/all-mpnet-base-v2'
  'MiniLM': 'sentence-transformers/all-MiniLM-L6-v2'
  
  # E5 Family
  'E5-Base': 'intfloat/e5-base'
  'E5-Large': 'intfloat/e5-large'

fields:
  # Computer & Information Science
  - "artificial intelligence"
  - "machine learning"
  - "computer vision"
  
  # Biological Sciences
  - "molecular biology"
  - "neuroscience"
  - "genomics"
  
  # Physical Sciences
  - "quantum physics"
  - "materials science"
  - "astronomical sciences"
```

## 🚀 Usage

### Basic Usage
```bash
python -m src.embedding_benchmarking.cli evaluate
```

### Advanced Options
```bash
python -m src.embedding_benchmarking.cli evaluate \
  --cache-dir embedding_cache \
  --max-papers 100 \
  --config custom_config.yaml \
  --output-dir results
```

## 📊 Output Files

1. `embedding_comparison_results.csv`: Detailed metrics
2. `model_leaderboard.csv`: Aggregated performance scores
3. Experiment directory with:
   - Leaderboard and detailed metrics comparisons 
   - Full statistical analysis
   - Configuration snapshots
   - Paper metadata
   - Detailed logs


## Project Structure

```
.
├── src/embedding_benchmarking/  # Main package source code
│   ├── __init__.py
│   ├── cli.py                  # Command line interface
│   ├── config.py               # Configuration handling
│   ├── data.py                 # Data processing
│   ├── embedding_evaluator.py  # Core evaluation logic
│   ├── evaluation.py           # Evaluation metrics
│   ├── models.py              # Model implementations
│   └── utils.py               # Utility functions
│
├── config/                     # Configuration files
│   ├── config-example.yaml    # Example configuration
│   └── .env-example          # Environment variables example
│
├── experiments/               # Experiment results
│   └── experiment_20241219_111718/
│
├── docs/                      # Documentation
│   ├── license.md
│
└── requirements.txt          # Project dependencies
```



## 📚 Citation

```bibtex
@software{embedding_benchmarking_arxiv,
  title = {Academic Embedding Model Evaluator},
  author = {Champion, Cody},
  year = {2024},
  description = {A tool for evaluating embedding models on academic paper similarity tasks}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Ready to revolutionize your research paper analysis? Get started today! 🚀
