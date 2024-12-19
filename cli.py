import click
import os
from dotenv import load_dotenv
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import pandas as pd

from utils import console, setup_logging
from config import Config
from models import ModelManager
from data import DataManager
from evaluation import Evaluator

@click.group()
def cli():
    """Academic Embedding Model Evaluator CLI."""
    pass

@click.command()
@click.option('--cache-dir', default='embedding_cache', help='Cache directory')
@click.option('--max-tokens', default=512, help='Maximum tokens in abstract')
@click.option('--min-tokens', default=50, help='Minimum tokens in abstract')
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Config file path')
def evaluate(cache_dir: str, max_tokens: int, min_tokens: int, config: str):
    """Run embedding model evaluation."""
    load_dotenv()
    setup_logging()
    
    if not os.getenv('HUGGINGFACE_TOKEN'):
        console.print("⚠️  [red]HUGGINGFACE_TOKEN not found in environment variables![/red]")
        console.print("Please set your HuggingFace token in the .env file")
        return

    console.print("\n[bold green]🚀 Starting Academic Embedding Model Evaluation[/bold green]")
    
    # Initialize components
    config_manager = Config(
        config_path=config,
        cache_dir=cache_dir,
        papers_per_field=25,  # Fixed at 25 papers per field
        max_tokens=max_tokens,
        min_tokens=min_tokens
    )
    
    model_manager = ModelManager(
        device=None,  # Will auto-detect
        hf_token=os.getenv('HUGGINGFACE_TOKEN')
    )
    
    data_manager = DataManager(config_manager, model_manager)
    evaluator = Evaluator(config_manager, model_manager)
    
    # Run evaluation pipeline
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        # Fetch papers
        papers = data_manager.fetch_papers(progress)
        
        if not papers:
            console.print("\n❌ Paper collection failed. Aborting comparison.")
            return
            
        if len(papers) == len(config_manager.fields) * config_manager.papers_per_field:
            evaluator.save_experiment_metadata(papers, Path('.'))
            
            results = []
            for model_name, model_path in config_manager.models.items():
                try:
                    # Create a single task for all comparisons
                    eval_task = progress.add_task(
                        "Starting comparisons...",
                        total=1  # Will be updated by evaluate_model
                    )
                    
                    scores = evaluator.evaluate_model(papers, model_path, progress, eval_task)
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
            results_df.to_csv('embedding_comparison_results.csv', index=False)
            evaluator.create_leaderboard(results_df)
        else:
            console.print("\n❌ Incorrect number of papers collected. Aborting comparison.")

cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
