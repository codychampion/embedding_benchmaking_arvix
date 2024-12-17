"""
Academic Embedding Model Evaluator CLI

A command-line tool for evaluating embedding models on academic paper similarity tasks.
"""

import arxiv
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import pickle
from pathlib import Path
import hashlib
import warnings
import logging
import os
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
import click
import yaml
import sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
from rich.style import Style

# Configure console for rich output
console = Console()

# Configure logging and warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress HuggingFace download messages
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._validators").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.hub_mixin").setLevel(logging.ERROR)

class EmbeddingEvaluator:
    """Evaluates embedding models on academic paper similarity tasks."""
    
    def __init__(self, cache_dir: str = 'embedding_cache', max_papers: int = 100, 
                 config_path: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            cache_dir: Directory to store embedding cache
            max_papers: Maximum number of papers per field
            config_path: Path to YAML config file for models and fields
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_papers = max_papers
        
        # Load configuration
        if config_path:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                self.models = config.get('models', {})
                self.fields = config.get('fields', [])
                console.print(f"📚 Loaded configuration from [cyan]{config_path}[/cyan]")
                console.print(f"🤖 Models to evaluate: [green]{len(self.models)}[/green]")
                console.print(f"🔬 Research fields: [green]{len(self.fields)}[/green]")
        else:
            # Default configurations
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
        """Generate cache file path for a specific text and model"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache_dir / f"{model_name}_{text_hash}.pkl"

    def get_embedding(self, text: str, model_name: str) -> np.ndarray:
        """Get embedding with caching"""
        model_name_split = model_name.split('/')[0]
        cache_path = self.get_cache_path(text, model_name_split)
        
        # Check cache first
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # Generate embedding if not cached
        if model_name == 'Bedrock':
            embedding = self._get_bedrock_embedding(text)
        else:
            embedding = self._get_hf_embedding(text, model_name)

        # Cache the result
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)

        return embedding

    def _get_bedrock_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Bedrock"""
        response = self.bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="*/*",
            body=json.dumps({
                "inputText": text,
                "dimensions": 512,
                "normalize": True
            })
        )
        return np.array(json.loads(response['body'].read())['embedding'])

    def _get_hf_embedding(self, text: str, model_name: str) -> np.ndarray:
        """Get embedding from HuggingFace model with suppressed outputs"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Suppress stdout during model loading
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv('HUGGINGFACE_TOKEN'))
                model = AutoModel.from_pretrained(model_name, token=os.getenv('HUGGINGFACE_TOKEN'))
            finally:
                sys.stdout = old_stdout

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            max_length = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 512
                
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)

            model = model.to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Normalize embedding
            embedding = embeddings[0].numpy()
            return embedding / np.linalg.norm(embedding)

    def evaluate_model(self, papers: List[Dict], model_name: str, progress: Progress, task_id: int) -> Dict:
        """Evaluate a model on papers"""
        results = {
            'title_abstract_same': [],
            'title_abstract_diff': [],
            'title_abstract_other': [],
            'abstract_abstract_same': [],
            'abstract_abstract_diff': []
        }
        
        total_steps = len(papers)
        for i, paper1 in enumerate(papers):
            # Update progress description
            progress.update(task_id, description=f"[cyan]Processing paper {i+1}/{total_steps} for {model_name}[/cyan]")
            
            # Get embeddings (will use cache if available)
            title_emb = self.get_embedding(paper1['title'], model_name)
            abstract1_emb = self.get_embedding(paper1['abstract'], model_name)
            
            # Title to own abstract
            sim = cosine_similarity([title_emb], [abstract1_emb])[0][0]
            results['title_abstract_same'].append(sim)
            
            for paper2 in papers:
                if paper1 != paper2:
                    abstract2_emb = self.get_embedding(paper2['abstract'], model_name)
                    
                    # Title to different abstract
                    sim_title_abs = cosine_similarity([title_emb], [abstract2_emb])[0][0]
                    if paper1['category'] == paper2['category']:
                        results['title_abstract_diff'].append(sim_title_abs)
                    else:
                        results['title_abstract_other'].append(sim_title_abs)
                    
                    # Abstract to abstract
                    sim_abs_abs = cosine_similarity([abstract1_emb], [abstract2_emb])[0][0]
                    if paper1['category'] == paper2['category']:
                        results['abstract_abstract_same'].append(sim_abs_abs)
                    else:
                        results['abstract_abstract_diff'].append(sim_abs_abs)
            
            progress.update(task_id, advance=1)

        return {k: (np.mean(v), np.std(v)) for k, v in results.items()}

    def run_comparison(self, output_dir: str = '.'):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Initialize progress display
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        )
        
        with progress:
            # Get papers
            papers = []
            papers_task = progress.add_task(
                "[cyan]Fetching papers[/cyan]",
                total=len(self.fields)
            )
            
            client = arxiv.Client()
            for query in self.fields:
                progress.update(papers_task, description=f"[cyan]Fetching papers for: {query}[/cyan]")
                search = arxiv.Search(
                    query=query,
                    max_results=self.max_papers,
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                
                for result in client.results(search):
                    papers.append({
                        'title': result.title,
                        'abstract': result.summary,
                        'category': result.primary_category
                    })
                progress.update(papers_task, advance=1)
            
            console.print(f"\n📊 Total papers collected: [green]{len(papers)}[/green]")
            
            # Run model evaluations
            results = []
            total_steps = len(papers) * len(self.models)
            eval_task = progress.add_task(
                "[cyan]Evaluating models[/cyan]",
                total=total_steps
            )
            
            for model_name, model_path in self.models.items():
                try:
                    progress.update(eval_task, description=f"[cyan]Evaluating: {model_name}[/cyan]")
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
        
        # Create results DataFrame and display
        df = pd.DataFrame(results)
        
        # Create a rich table for results
        table = Table(title="Model Comparison Results", show_header=True, header_style="bold magenta")
        for column in df.columns:
            table.add_column(column)
        for _, row in df.iterrows():
            table.add_row(*[str(val) for val in row])
        
        console.print("\n")
        console.print(Panel(table, title="📊 Results", border_style="green"))
        
        # Save results
        results_path = output_dir / 'embedding_comparison_results.csv'
        df.to_csv(results_path, index=False)
        console.print(f"\n💾 Results saved to: [cyan]{results_path}[/cyan]")

        leaderboard = self.create_leaderboard(df)

    def create_leaderboard(self, results_df):
        """Create a leaderboard with performance metrics"""
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

        # Calculate scores for each model
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

        # Create leaderboard
        leaderboard = pd.DataFrame(scores)
        leaderboard = leaderboard.sort_values("Score", ascending=False)

        # Create a rich table for the leaderboard
        table = Table(title="🏆 Model Leaderboard", show_header=True, header_style="bold magenta")
        for column in leaderboard.columns:
            table.add_column(column)
        
        # Add rows with alternating colors
        for i, (_, row) in enumerate(leaderboard.iterrows()):
            style = "green" if i == 0 else "white"
            table.add_row(*[str(val) for val in row], style=style)

        console.print("\n")
        console.print(Panel(table, border_style="green"))

        # Save leaderboard
        leaderboard.to_csv("model_leaderboard.csv", index=False)
        console.print(f"\n💾 Leaderboard saved to: [cyan]model_leaderboard.csv[/cyan]")

        return leaderboard


@click.group()
def cli():
    """Academic Embedding Model Evaluator - Compare embedding models on academic papers."""
    pass

@cli.command()
@click.option('--cache-dir', default='embedding_cache', 
              help='Directory to store embedding cache')
@click.option('--max-papers', default=3, 
              help='Maximum number of papers per field')
@click.option('--config', default='config.yaml', type=click.Path(exists=True), 
              help='Path to YAML config file for models and fields')
@click.option('--output-dir', default='.',
              help='Directory to save results')
def evaluate(cache_dir: str, max_papers: int, config: Optional[str], output_dir: str):
    """Run embedding model evaluation."""
    # Load environment variables
    load_dotenv()
    
    # Check for HuggingFace token
    if not os.getenv('HUGGINGFACE_TOKEN'):
        console.print("⚠️  [red]HUGGINGFACE_TOKEN not found in environment variables![/red]")
        console.print("Please set your HuggingFace token in the .env file")
        return

    console.print("\n[bold green]🚀 Starting Academic Embedding Model Evaluation[/bold green]")
    console.print("=" * 50)
    
    evaluator = EmbeddingEvaluator(cache_dir=cache_dir, max_papers=max_papers, config_path=config)
    evaluator.run_comparison(output_dir=output_dir)
    
    console.print("\n[bold green]✨ Evaluation complete![/bold green]")
    console.print("=" * 50)

@cli.command()
@click.argument('output', type=click.Path())
def init_config(output: str):
    """Generate a template configuration file."""
    config = {
        'models': {
            'allenai/specter': 'allenai/specter',
            'allenai/scibert_scivocab_uncased': 'allenai/scibert_scivocab_uncased',
            'sentence-transformers/all-mpnet-base-v2': 'sentence-transformers/all-mpnet-base-v2',
        },
        'fields': [
            "artificial intelligence",
            "machine learning",
            "computer vision",
            "natural language processing",
        ]
    }
    
    with open(output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    console.print(f"✨ Configuration template generated at: [cyan]{output}[/cyan]")

if __name__ == "__main__":
    cli()
