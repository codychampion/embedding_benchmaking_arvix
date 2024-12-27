# 🏆 Academic Embedding Model Evaluator

**Author**: Cody Champion

## Current Model Leaderboard

Our sophisticated leaderboard system shows exactly how different AI models perform at understanding research papers:

```
╭────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                    🏆 Model Leaderboard                                         │
│ ┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓        │
│ ┃ Model             ┃ Score ┃ Own Title-Abstract ┃ Same Field Separation ┃ Avg Std ┃        │
│ ┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩        │
│ │ Bedrock          │ 0.449 │ 0.710              │ 0.103                 │ 0.118   │        │
│ │ MPNet            │ 0.443 │ 0.714              │ 0.271                 │ 0.134   │        │
│ │ MiniLM-L12       │ 0.439 │ 0.688              │ 0.246                 │ 0.130   │        │
│ │ MiniLM-L6        │ 0.433 │ 0.667              │ 0.242                 │ 0.129   │        │
│ │ RoBERTa-Large-ST │ 0.410 │ 0.601              │ 0.165                 │ 0.110   │        │
│ │ DistilRoBERTa    │ 0.410 │ 0.593              │ 0.185                 │ 0.120   │        │
│ │ MS-Marco         │ 0.374 │ 0.582              │ 0.135                 │ 0.126   │        │
│ │ S-BioBert        │ 0.352 │ 0.637              │ 0.163                 │ 0.118   │        │
│ │ Contriever       │ 0.344 │ 0.664              │ 0.108                 │ 0.082   │        │
│ │ Multi-MiniLM     │ 0.343 │ 0.495              │ 0.135                 │ 0.130   │        │
│ │ QA-MPNet         │ 0.332 │ 0.673              │ 0.092                 │ 0.078   │        │
│ │ BioBERT-NLI      │ 0.302 │ 0.608              │ 0.097                 │ 0.105   │        │
│ │ Specter          │ 0.278 │ 0.788              │ 0.086                 │ 0.072   │        │
│ │ GTE-Large        │ 0.274 │ 0.926              │ 0.069                 │ 0.038   │        │
│ │ GTE-Base         │ 0.273 │ 0.919              │ 0.066                 │ 0.038   │        │
│ │ BGE-Large        │ 0.266 │ 0.925              │ 0.074                 │ 0.036   │        │
│ │ E5-Large         │ 0.265 │ 0.890              │ 0.052                 │ 0.034   │        │
│ │ E5-Base          │ 0.262 │ 0.848              │ 0.045                 │ 0.036   │        │
│ │ BGE-Base         │ 0.262 │ 0.887              │ 0.064                 │ 0.034   │        │
│ │ SciBERT          │ 0.256 │ 0.714              │ 0.049                 │ 0.052   │        │
│ │ BERT-Base        │ 0.249 │ 0.779              │ 0.056                 │ 0.060   │        │
│ │ BiomedVLP        │ 0.243 │ 0.778              │ 0.045                 │ 0.051   │        │
│ │ DistilBERT       │ 0.239 │ 0.837              │ 0.045                 │ 0.046   │        │
│ │ BERT-Large       │ 0.238 │ 0.845              │ 0.045                 │ 0.046   │        │
│ │ BioLinkBERT      │ 0.235 │ 0.851              │ 0.043                 │ 0.041   │        │
│ │ BioBERT          │ 0.223 │ 0.886              │ 0.025                 │ 0.029   │        │
│ │ BiomedNLP        │ 0.208 │ 0.962              │ 0.008                 │ 0.011   │        │
│ │ RoBERTa-Base     │ 0.208 │ 0.926              │ 0.007                 │ 0.016   │        │
│ └─────────────────┴───────┴────────────────────┴───────────────────────┴─────────┘        │
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
📚 Loaded configuration from config/config.yaml
🤖 Models to evaluate: 24
🔬 Research fields: 35
📄 Papers per field: 50

Fetching papers...

📊 Query statistics for artificial intelligence:
   - Papers found: 50
   - Papers checked: 50
   - Rejected (too short): 0
   - Rejected (too long): 0

📊 Query statistics for machine learning:
   - Papers found: 50
   - Papers checked: 50
   - Rejected (too short): 0
   - Rejected (too long): 0

....

📊 Paper Collection Summary:
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Field                          ┃ Papers ┃ Avg Tokens ┃ Token Range ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ artificial intelligence        │ 50     │ 250.0      │ 88-451      │
│ machine learning               │ 50     │ 254.9      │ 73-472      │
│ computer vision                │ 50     │ 250.0      │ 65-468      │
│ natural language processing    │ 50     │ 254.6      │ 65-502      │
│ cybersecurity                  │ 50     │ 250.1      │ 67-488      │
│ quantum computing              │ 50     │ 245.5      │ 65-483      │
│ ....                           │ ...    │ ...        │ ...         │
│ advanced manufacturing         │ 50     │ 254.0      │ 76-502      │
│ biotechnology                  │ 50     │ 243.8      │ 54-502      │
│ computational biology          │ 50     │ 249.6      │ 65-468      │
│ quantum materials              │ 50     │ 238.8      │ 54-502      │
│ sustainable energy             │ 50     │ 247.9      │ 57-502      │
│ artificial intelligence ethics │ 50     │ 250.1      │ 88-451      │
└────────────────────────────────┴────────┴────────────┴─────────────┘

✅ Processed: BERT-Large
✅ Processed: RoBERTa-Base
...

⠦    Processing Models (15/24) - Est. 2.1h remaining ━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━  62%      
Getting title embeddings (50/1750)   ━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  11%      0:01:31   

Saving results...


╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                🏆 Model Leaderboard                                                                                                                                        │
│ ┏━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓                                                                                                        │
│ ┃ Model            ┃ Score ┃ Own Title-Abstract ┃ Same Field Separation ┃ Avg Std ┃                                                                                                        │
│ ┡━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩                                                                                                        │
│ │ Bedrock          │ 0.449 │ 0.710              │ 0.103                 │ 0.118   │                                                                                                        │
│ │ MPNet            │ 0.443 │ 0.714              │ 0.271                 │ 0.134   │                                                                                                        │
│ │ MiniLM-L12       │ 0.439 │ 0.688              │ 0.246                 │ 0.130   │                                                                                                        │
│ │ MiniLM-L6        │ 0.433 │ 0.667              │ 0.242                 │ 0.129   │                                                                                                        │
│ │ ...              │ ...   │ ...                │ ...                   │ ...     │
│ │ SciBERT          │ 0.256 │ 0.714              │ 0.049                 │ 0.052   │                                                                                                        │
│ │ BERT-Base        │ 0.249 │ 0.779              │ 0.056                 │ 0.060   │                                                                                                        │
│ │ BiomedVLP        │ 0.243 │ 0.778              │ 0.045                 │ 0.051   │                                                                                                        │
│ │ DistilBERT       │ 0.239 │ 0.837              │ 0.045                 │ 0.046   │                                                                                                        │
│ │ BERT-Large       │ 0.238 │ 0.845              │ 0.045                 │ 0.046   │                                                                                                        │
│ │ BioLinkBERT      │ 0.235 │ 0.851              │ 0.043                 │ 0.041   │                                                                                                        │
│ │ BioBERT          │ 0.223 │ 0.886              │ 0.025                 │ 0.029   │                                                                                                        │
│ │ BiomedNLP        │ 0.208 │ 0.962              │ 0.008                 │ 0.011   │                                                                                                        │
│ │ RoBERTa-Base     │ 0.208 │ 0.926              │ 0.007                 │ 0.016   │                                                                                                        │
│ └──────────────────┴───────┴────────────────────┴───────────────────────┴─────────┘                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

💾 Leaderboard saved to: experiments\experiment_20241226_112605\model_leaderboard.csv

```

## 📊 Professional Experiment Tracking

Every experiment is automatically documented with detailed statistics and configurations:

### 📈 Collection Statistics
```yaml
# collection_statistics.yaml
date_range:
  earliest: '2024-03-18'
  latest: '2024-12-20'
papers_per_field:
  advanced manufacturing: 50
  artificial intelligence: 50
  artificial intelligence ethics: 50
  astronomical sciences: 50
  atmospheric science: 50
  bioinformatics: 50
  biomedical engineering: 50
  biotechnology: 50
  climate science: 50
  cognitive science: 50
  computational biology: 50
  computational social science: 50
  computer vision: 50
  condensed matter physics: 50
  cybersecurity: 50
  data science: 50
  environmental science: 50
  genomics: 50
  machine learning: 50
  materials chemistry: 50
  materials science: 50
  molecular biology: 50
  nanotechnology: 50
  natural language processing: 50
  neuroscience: 50
  particle physics: 50
  quantum computing: 50
  quantum information science: 50
  quantum materials: 50
  quantum physics: 50
  renewable energy: 50
  robotics: 50
  semiconductor physics: 50
  sustainable energy: 50
  synthetic biology: 50
token_statistics:
  max: 502
  mean: 250.04057142857144
  median: 246.0
  min: 53
  std: 76.35648136649283
total_papers: 1750

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

Create your config file with our comprehensive model selection (there are many more in the example yaml):

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
