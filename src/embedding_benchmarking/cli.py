import click
import os
from dotenv import load_dotenv
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
import pandas as pd
from datetime import datetime
from .utils import console, setup_logging
from .config import Config
from .models import ModelManager
from .data import DataManager
from .evaluation import Evaluator

@click.group()
def cli():
    """Academic Embedding Model Evaluator CLI."""
    pass

@click.command()
@click.option('--cache-dir', default='embedding_cache', help='Cache directory')
@click.option('--max-tokens', default=512, help='Maximum tokens in abstract')
@click.option('--papers_per_field', default=5, help='Maximum papers per field')
@click.option('--min-tokens', default=50, help='Minimum tokens in abstract')
@click.option('--config', default='config/config.yaml', type=click.Path(exists=True), help='Config file path')
def evaluate(cache_dir: str, max_tokens: int, papers_per_field: int, min_tokens: int, config: str):
    """Run embedding model evaluation."""
    load_dotenv(), 
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
        papers_per_field=papers_per_field,
        max_tokens=max_tokens,
        min_tokens=min_tokens
    )
    
    model_manager = ModelManager(
        device=None,  # Will auto-detect
        hf_token=os.getenv('HUGGINGFACE_TOKEN')
    )
    
    data_manager = DataManager(config_manager, model_manager)
    evaluator = Evaluator(config_manager, model_manager)
    
    # Fetch papers (without progress bar)
    console.print("\n[yellow]Fetching papers...[/yellow]")
    papers = data_manager.fetch_papers()
    
    if not papers:
        console.print("\n❌ Paper collection failed. Aborting comparison.")
        return
        
    if len(papers) == len(config_manager.fields) * config_manager.papers_per_field:
        # Create experiment directory and save metadata
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = Path('.') / f'experiments/experiment_{timestamp}'
        
        console.print("\n[yellow]Saving experiment metadata...[/yellow]")
        evaluator.save_experiment_metadata(papers, experiment_dir)
        
        # Single progress bar just for model evaluation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True,
            transient=False,
            refresh_per_second=5  # Reduced refresh rate for stability
        ) as progress:
            # One task for all models
            task = progress.add_task(
                "[cyan]Evaluating models...", 
                total=len(config_manager.models)
            )
            
            results = []
            for model_name, model_path in config_manager.models.items():
                try:
                    progress.update(task, description=f"[cyan]Evaluating {model_name}[/cyan]")
                    
                    scores = evaluator.evaluate_model(
                        papers=papers,
                        model_name=model_path,
                        #display_name=model_name
                    )
                    
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
                    progress.advance(task)
                    console.print(f"✅ Processed: [green]{model_name}[/green]")

                except Exception as e:
                    console.print(f"❌ Failed: [red]{model_name}[/red] - {str(e)}")
        
        # Save results
        console.print("\n[yellow]Saving results...[/yellow]")
        results_df = pd.DataFrame(results)
        results_df.to_csv(experiment_dir / 'embedding_comparison_results.csv', index=False)
        evaluator.create_leaderboard(results_df, experiment_dir)
    else:
        console.print("\n❌ Incorrect number of papers collected. Aborting comparison.")

cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
