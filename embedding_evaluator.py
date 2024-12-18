import arxiv
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import pickle
from pathlib import Path
import hashlib
import warnings
import logging
import os
import shutil
from typing import List, Dict, Tuple, Optional, Any, Union
from dotenv import load_dotenv
import click
import yaml
import sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
from rich.style import Style
from datetime import datetime, timedelta

# Configure console for rich output
console = Console()

# Configure logging and warnings
warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress HuggingFace download messages
for logger in ["huggingface_hub.file_download", 
              "huggingface_hub.utils._validators",
              "huggingface_hub.hub_mixin"]:
    logging.getLogger(logger).setLevel(logging.ERROR)

class EmbeddingEvaluator:
    """Evaluates embedding models on academic paper similarity tasks."""
    
    def __init__(self, 
                 cache_dir: str = 'embedding_cache',
                 papers_per_field: int = 25,
                 max_tokens: int = 512,
                 min_tokens: int = 50,
                 config_path: Optional[str] = None,
                 device: Optional[str] = None):
        """Initialize the evaluator."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.papers_per_field = papers_per_field
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize fast tokenizer for length checking
        console.print("🔄 Initializing length checking tokenizer...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.length_tokenizer = AutoTokenizer.from_pretrained(
                'bert-base-uncased',
                token=os.getenv('HUGGINGFACE_TOKEN')
            )
        
        # Model cache
        self._model_cache = {}
        self._tokenizer_cache = {}
        
        self._load_config(config_path)

    def _load_config(self, config_path: Optional[str]) -> None:
        """Load configuration from file or use defaults."""
        if config_path:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                self.models = config.get('models', {})
                self.fields = config.get('fields', [])
                console.print(f"📚 Loaded configuration from [cyan]{config_path}[/cyan]")
                console.print(f"🤖 Models to evaluate: [green]{len(self.models)}[/green]")
                console.print(f"🔬 Research fields: [green]{len(self.fields)}[/green]")
        else:
            self.models = {
                'allenai/specter': 'allenai/specter',
                'allenai/scibert_scivocab_uncased': 'allenai/scibert_scivocab_uncased',
                'sentence-transformers/all-mpnet-base-v2': 'sentence-transformers/all-mpnet-base-v2',
            }
            self.fields = [
                "artificial intelligence",
                "machine learning",
                "computer vision",
            ]
            console.print("ℹ️ Using default configuration")

    def get_cache_path(self, text: str, model_name: str) -> Path:
        """Generate cache file path for a specific text and model."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        model_name_safe = model_name.replace('/', '_')
        return self.cache_dir / f"{model_name_safe}_{text_hash}.pkl"

    def _check_text_length(self, text: str) -> Tuple[bool, int]:
        """Check if text meets length requirements."""
        tokens = self.length_tokenizer(text, truncation=False)['input_ids']
        token_count = len(tokens)
        return (self.min_tokens <= token_count <= self.max_tokens, token_count)

    def _get_model_and_tokenizer(self, model_name: str) -> Tuple[Any, Any]:
        """Get or load model and tokenizer with caching."""
        if model_name not in self._model_cache:
            with warnings.catch_warnings(), open(os.devnull, 'w') as f:
                old_stdout = sys.stdout
                sys.stdout = f
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        token=os.getenv('HUGGINGFACE_TOKEN')
                    )
                    model = AutoModel.from_pretrained(
                        model_name,
                        token=os.getenv('HUGGINGFACE_TOKEN')
                    ).to(self.device)
                    model.eval()
                finally:
                    sys.stdout = old_stdout

            self._model_cache[model_name] = model
            self._tokenizer_cache[model_name] = tokenizer

        return self._model_cache[model_name], self._tokenizer_cache[model_name]

    def get_embeddings_batch(self, 
                           texts: List[str], 
                           model_name: str,
                           show_progress: bool = False) -> np.ndarray:
        """Get embeddings for a batch of texts."""
        cache_hits = []
        cache_misses = []
        cache_miss_indices = []
        
        for i, text in enumerate(texts):
            cache_path = self.get_cache_path(text, model_name)
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cache_hits.append(pickle.load(f))
            else:
                cache_misses.append(text)
                cache_miss_indices.append(i)
        
        if not cache_misses:
            return np.stack(cache_hits)
        
        model, tokenizer = self._get_model_and_tokenizer(model_name)
        inputs = tokenizer(
            cache_misses,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            outputs = model(**inputs)
        
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        sum_embeddings = torch.einsum(
            "bsh,bs->bh",
            token_embeddings,
            attention_mask.float()
        )
        mask_sums = attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-9)
        embeddings = sum_embeddings / mask_sums
        
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.cpu().numpy()
        
        for i, (idx, embedding) in enumerate(zip(cache_miss_indices, embeddings)):
            cache_path = self.get_cache_path(cache_misses[i], model_name)
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
            cache_hits.insert(idx, embedding)
        
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        return np.stack(cache_hits)

    def _fetch_papers_for_field(self, 
                              query: str, 
                              progress: Progress,
                              task_id: int,
                              max_attempts: int = 1000) -> List[Dict]:
        """Fetch papers for a specific field."""
        field_papers = []
        attempts = 0
        start_date = datetime.now() - timedelta(days=365 * 5)
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=f"{query} AND submittedDate:[{start_date.strftime('%Y%m%d')}0000 TO *]",
            max_results=max_attempts,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        for result in client.results(search):
            attempts += 1
            meets_length, token_count = self._check_text_length(result.summary)
            
            if meets_length:
                paper = {
                    'title': result.title,
                    'abstract': result.summary,
                    'token_count': token_count,
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'primary_category': result.primary_category,
                    'categories': ','.join(result.categories),
                    'query_field': query,
                    'published_date': result.published.strftime('%Y-%m-%d'),
                    'updated_date': result.updated.strftime('%Y-%m-%d') if result.updated else None,
                    'authors': ','.join([author.name for author in result.authors]),
                    'author_count': len(result.authors),
                    'collection_timestamp': datetime.now().isoformat(),
                    'title_token_count': len(self.length_tokenizer(result.title)['input_ids']),
                    'abstract_token_count': token_count,
                }
                
                field_papers.append(paper)
                
                progress.update(
                    task_id,
                    advance=1,
                    description=f"[cyan]{query}: {len(field_papers)}/{self.papers_per_field} papers"
                )
                
                if len(field_papers) >= self.papers_per_field:
                    break
            
            if attempts >= max_attempts:
                raise ValueError(
                    f"Could not find {self.papers_per_field} papers for field '{query}' "
                    f"after checking {max_attempts} papers. Found {len(field_papers)}."
                )
        
        return field_papers[:self.papers_per_field]

    def _fetch_papers(self, progress: Progress) -> List[Dict]:
        """Fetch all papers."""
        papers = []
        papers_task = progress.add_task(
            "[cyan]Fetching papers[/cyan]",
            total=len(self.fields) * self.papers_per_field
        )
        
        for query in self.fields:
            try:
                field_papers = self._fetch_papers_for_field(
                    query, 
                    progress,
                    papers_task
                )
                papers.extend(field_papers)
                
            except ValueError as e:
                console.print(f"\n⚠️ Warning for field '{query}': {str(e)}")
                return []
        
        console.print("\n📊 Paper Collection Summary:")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Field")
        table.add_column("Papers")
        table.add_column("Avg Tokens")
        table.add_column("Token Range")
        
        for field in self.fields:
            field_papers = [p for p in papers if p['query_field'] == field]
            token_counts = [p['token_count'] for p in field_papers]
            
            table.add_row(
                field,
                str(len(field_papers)),
                f"{np.mean(token_counts):.1f}",
                f"{min(token_counts)}-{max(token_counts)}"
            )
        
        console.print(table)
        
        if len(papers) != len(self.fields) * self.papers_per_field:
            console.print(f"\n❌ Error: Got {len(papers)} papers, "
                        f"expected {len(self.fields) * self.papers_per_field}")
            return []
        
        console.print(f"\n✅ Successfully collected {len(papers)} papers "
                     f"({self.papers_per_field} per field)")
        
        return papers

    def evaluate_model(self, 
                      papers: List[Dict], 
                      model_name: str, 
                      progress: Progress, 
                      task_id: int) -> Dict[str, Tuple[float, float]]:
        """Evaluate a model on papers."""
        results = {
            'title_abstract_same': [],
            'title_abstract_diff': [],
            'title_abstract_other': [],
            'abstract_abstract_same': [],
            'abstract_abstract_diff': []
        }
        
        titles = [p['title'] for p in papers]
        abstracts = [p['abstract'] for p in papers]
        
        progress.update(task_id, description=f"[cyan]Computing embeddings for {model_name}[/cyan]")
        
        title_embeddings = self.get_embeddings_batch(titles, model_name)
        abstract_embeddings = self.get_embeddings_batch(abstracts, model_name)
        
        title_abstract_sims = cosine_similarity(title_embeddings, abstract_embeddings)
        abstract_abstract_sims = cosine_similarity(abstract_embeddings)
        
        for i, paper1 in enumerate(papers):
            results['title_abstract_same'].append(title_abstract_sims[i, i])
            
            for j, paper2 in enumerate(papers):
                if i != j:
                    if paper1['query_field'] == paper2['query_field']:
                        results['title_abstract_diff'].append(title_abstract_sims[i, j])
                        results['abstract_abstract_same'].append(abstract_abstract_sims[i, j])
                    else:
                        results['title_abstract_other'].append(title_abstract_sims[i, j])
                        results['abstract_abstract_diff'].append(abstract_abstract_sims[i, j])
        
        progress.update(task_id, advance=len(papers))
        
        return {k: (float(np.mean(v)), float(np.std(v))) for k, v in results.items()}

    def save_experiment_metadata(self, 
                               papers: List[Dict], 
                               output_dir: Path,
                               include_embeddings: bool = True) -> None:
        """Save experiment metadata."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = output_dir / f'experiment_{timestamp}'
        experiment_dir.mkdir(exist_ok=True)
        
        config = {
            'timestamp': datetime.now().isoformat(),
            'papers_per_field': self.papers_per_field,
            'max_tokens': self.max_tokens,
            'min_tokens': self.min_tokens,
            'fields': self.fields,
            'models': self.models,
            'device': self.device,
            'python_version': sys.version,
            'package_versions': {
                'torch': torch.__version__,
                'transformers': transformers.__version__,
                'numpy': np.__version__,
                'pandas': pd.__version__,
                'arxiv': arxiv.__version__
            }
        }
        
        with open(experiment_dir / 'experiment_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        paper_df = pd.DataFrame(papers)
        paper_df.to_csv(experiment_dir / 'papers_full.csv', index=False)
        clean_df = paper_df.drop(columns=['abstract'])
        clean_df.to_csv(experiment_dir / 'papers_metadata.csv', index=False)
        
        stats = {
            'total_papers': len(papers),
            'papers_per_field': {
                field: len([p for p in papers if p['query_field'] == field])
                for field in self.fields
            },
            'token_statistics': {
                'mean': paper_df['token_count'].mean(),
                'median': paper_df['token_count'].median(),
                'min': paper_df['token_count'].min(),
                'max': paper_df['token_count'].max(),
                'std': paper_df['token_count'].std()
            },
            'date_range': {
                'earliest': paper_df['published_date'].min(),
                'latest': paper_df['published_date'].max()
            }
        }
        
        with open(experiment_dir / 'collection_statistics.yaml', 'w') as f:
            yaml.dump(stats, f, default_flow_style=False)
        
        readme_content = f"""# Embedding Evaluation Experiment

## Experiment Details
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Papers per field: {self.papers_per_field}
- Total papers: {len(papers)}
- Token range: {self.min_tokens}-{self.max_tokens}

## Fields
{chr(10).join(f'- {field}' for field in self.fields)}

## Models
{chr(10).join(f'- {model}' for model in self.models.keys())}

## Files
- `experiment_config.yaml`: Full experiment configuration
- `papers_full.csv`: Complete paper data including abstracts
- `papers_metadata.csv`: Paper metadata (excluding abstracts)
- `collection_statistics.yaml`: Statistical summary of the collection
- `embeddings/`: Cached embeddings (if included)

## Reproduction
To reproduce this experiment:
1. Use the same configuration from `experiment_config.yaml`
2. Ensure the same package versions are installed
3. Run with the same papers from `papers_full.csv`
"""
        
        with open(experiment_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        if include_embeddings and self.cache_dir.exists():
            embeddings_dir = experiment_dir / 'embeddings'
            embeddings_dir.mkdir(exist_ok=True)
            for cache_file in self.cache_dir.glob('*.pkl'):
                shutil.copy2(cache_file, embeddings_dir / cache_file.name)
        
        console.print(f"\n💾 Experiment metadata saved to: [cyan]{experiment_dir}[/cyan]")

    def create_leaderboard(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Create a leaderboard ranking models."""
        def calculate_score(row):
            good_signals = [
                float(row["Title-Own Abstract Mean"]),
                float(row["Abstract-Abstract (Same Field) Mean"]),
                1 - float(row["Title-Diff Abstract (Diff Field) Mean"]),
                1 - float(row["Abstract-Abstract (Diff Field) Mean"]),
                1 - float(row["Title-Diff Abstract (Same Field) Mean"]),
            ]

            std_penalties = [
                -float(row["Title-Own Abstract Std"]),
                -float(row["Abstract-Abstract (Same Field) Std"]),
                -float(row["Title-Diff Abstract (Diff Field) Std"]),
                -float(row["Abstract-Abstract (Diff Field) Std"]),
                -float(row["Title-Diff Abstract (Same Field) Std"]),
            ]

            separations = [
                float(row["Title-Own Abstract Mean"]) - float(row["Title-Diff Abstract (Same Field) Mean"]),
                float(row["Abstract-Abstract (Same Field) Mean"]) - float(row["Abstract-Abstract (Diff Field) Mean"]),
            ]

            signal_weight = 0.5
            std_weight = 0.2
            separation_weight = 0.3

            final_score = (
                signal_weight * np.mean(good_signals)
                + std_weight * np.mean(std_penalties)
                + separation_weight * np.mean(separations)
            )

            return final_score

        scores = []
        for _, row in results_df.iterrows():
            model_name = row["Model"]
            score = calculate_score(row)

            key_metrics = {
                "Score": f"{score:.3f}",
                "Own Title-Abstract": f"{float(row['Title-Own Abstract Mean']):.3f}",
                "Same Field Separation": f"{(float(row['Abstract-Abstract (Same Field) Mean']) - float(row['Abstract-Abstract (Diff Field) Mean'])):.3f}",
                "Avg Std": f"{np.mean([float(row[col]) for col in row.index if 'Std' in col]):.3f}",
            }

            scores.append({"Model": model_name, **key_metrics})

        leaderboard = pd.DataFrame(scores)
        leaderboard = leaderboard.sort_values("Score", ascending=False)

        table = Table(title="🏆 Model Leaderboard", show_header=True, header_style="bold magenta")
        for column in leaderboard.columns:
            table.add_column(column)
        
        for i, (_, row) in enumerate(leaderboard.iterrows()):
            style = "green" if i == 0 else "white"
            table.add_row(*[str(val) for val in row], style=style)

        console.print("\n")
        console.print(Panel(table, border_style="green"))
        leaderboard.to_csv("model_leaderboard.csv", index=False)
        console.print(f"\n💾 Leaderboard saved to: [cyan]model_leaderboard.csv[/cyan]")

        return leaderboard

    def run_comparison(self, output_dir: str = '.') -> None:
        """Run the complete comparison pipeline."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            papers = self._fetch_papers(progress)
            
            if not papers:
                console.print("\n❌ Paper collection failed. Aborting comparison.")
                return
                
            if len(papers) == len(self.fields) * self.papers_per_field:
                self.save_experiment_metadata(papers, output_dir)
                
                results = []
                total_comparisons = len(papers) * len(self.models)
                eval_task = progress.add_task(
                    "[cyan]Evaluating models[/cyan]",
                    total=total_comparisons
                )
                
                for model_name, model_path in self.models.items():
                    try:
                        scores = self.evaluate_model(papers, model_path, progress, eval_task)
                        results.append({
                            'Model': model_name,
                            'Title-Own Abstract Mean': f"{scores['title_abstract_same'][0]:.3f}",
                            'Title-Own Abstract Std': f"{scores['title_abstract_same'][1]:.3f}",
                            'Title-Diff Abstract (Same Field) Mean': f"{scores['title_abstract_diff'][0]:.3f}",
                            'Title-Diff Abstract (Same Field) Std': f"{scores['title_abstract_diff'][1]:.3f}",
                            'Title-Diff Abstract (Diff Field) Mean': f"{scores['title_abstract_other'][0]:.3f}",
                            'Title-Diff Abstract (Diff Field) Std': f"{scores['title_abstract_other'][1]:.3f}",
                            'Abstract-Abstract (Same Field) Mean': f"{scores['abstract_abstract_same'][0]:.3f}",
                            'Abstract-Abstract (Same Field) Std': f"{scores['abstract_abstract_same'][1]:.3f}",
                            'Abstract-Abstract (Diff Field) Mean': f"{scores['abstract_abstract_diff'][0]:.3f}",
                            'Abstract-Abstract (Diff Field) Std': f"{scores['abstract_abstract_diff'][1]:.3f}"
                        })
                    except Exception as e:
                        console.print(f"❌ Failed: [red]{model_name}[/red] - {str(e)}")
                
                results_df = pd.DataFrame(results)
                results_df.to_csv(output_dir / 'embedding_comparison_results.csv', index=False)
                self.create_leaderboard(results_df)
            else:
                console.print("\n❌ Incorrect number of papers collected. Aborting comparison.")

@click.group()
def cli():
    """Academic Embedding Model Evaluator CLI."""
    pass

@click.command()
@click.option('--cache-dir', default='embedding_cache', help='Cache directory')
@click.option('--max-tokens', default=512, help='Maximum tokens in abstract')
@click.option('--min-tokens', default=50, help='Minimum tokens in abstract')
@click.option('--config', default=None, type=click.Path(exists=True), help='Config file path')
def evaluate(cache_dir: str, max_tokens: int, min_tokens: int, config: Optional[str]):
    """Run embedding model evaluation."""
    load_dotenv()
    
    if not os.getenv('HUGGINGFACE_TOKEN'):
        console.print("⚠️  [red]HUGGINGFACE_TOKEN not found in environment variables![/red]")
        console.print("Please set your HuggingFace token in the .env file")
        return

    console.print("\n[bold green]🚀 Starting Academic Embedding Model Evaluation[/bold green]")
    
    evaluator = EmbeddingEvaluator(
        cache_dir=cache_dir,
        papers_per_field=25,  # Fixed at 25 papers per field
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        config_path=config
    )
    evaluator.run_comparison()

cli.add_command(evaluate)

if __name__ == "__main__":
    cli()                                
