# Embedding Evaluation Experiment

## Experiment Details
- Date: 2024-12-26 11:26:05
- Papers per field: 100
- Total papers: 3500
- Token range: 50-512

## Fields
- artificial intelligence
- machine learning
- computer vision
- natural language processing
- cybersecurity
- quantum computing
- robotics
- biomedical engineering
- nanotechnology
- materials science
- semiconductor physics
- renewable energy
- quantum physics
- particle physics
- condensed matter physics
- materials chemistry
- astronomical sciences
- molecular biology
- neuroscience
- genomics
- synthetic biology
- bioinformatics
- computational social science
- cognitive science
- data science
- climate science
- atmospheric science
- environmental science
- quantum information science
- advanced manufacturing
- biotechnology
- computational biology
- quantum materials
- sustainable energy
- artificial intelligence ethics

## Models
- Bedrock
- Specter
- SciBERT
- BioLinkBERT
- BiomedNLP
- S-BioBert
- BioBERT-NLI
- MPNet
- MiniLM-L6
- QA-MPNet
- DistilRoBERTa
- RoBERTa-Large-ST
- MS-Marco
- MiniLM-L12
- Multi-MiniLM
- E5-Base
- E5-Large
- BERT-Base
- BERT-Large
- RoBERTa-Base
- RoBERTa-Large
- DistilBERT
- DeBERTa-V3-Base
- DeBERTa-V3-Large
- BGE-Base
- BGE-Large
- SciBERT-Rerank
- BioBERT
- BiomedVLP
- Longformer
- GTE-Base
- GTE-Large
- DeBERTa-MNLI
- Contriever

## Files
- `experiment_config.yaml`: Full experiment configuration
- `papers_full.csv`: Complete paper data including abstracts
- `papers_metadata.csv`: Paper metadata (excluding abstracts)
- `collection_statistics.yaml`: Statistical summary of the collection
- `embedding_comparison_results.csv`: Detailed model comparison metrics
- `model_leaderboard.csv`: Final model rankings and scores

## Implementation Notes
- Date handling: Published dates are stored as strings in the format 'YYYY-MM-DD' and converted to datetime objects when needed for statistics
- Removed embeddings directory as it's no longer needed

## Reproduction
To reproduce this experiment:
1. Use the same configuration from `experiment_config.yaml`
2. Ensure the same package versions are installed
3. Run with the same papers from `papers_full.csv`
