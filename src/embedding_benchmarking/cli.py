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
@click.option('--min-tokens', default=50, help='Minimum tokens in abstract')
@click.option('--config', default='config/config.yaml', type=click.Path(exists=True), help='Config file path')
def evaluate(cache_dir: str, max_tokens: int, min_tokens: int, config: str):
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
        
    # Create experiment directory and save metadata
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = Path('.') / f'experiments/experiment_{timestamp}'
    
    console.print("\n[yellow]Saving experiment metadata...[/yellow]")
    evaluator.save_experiment_metadata(papers, experiment_dir)
    
    # Single progress bar just for model evaluation
    from time import time
    
    def format_time(seconds: float) -> str:
        """Format time in appropriate units."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
        transient=False,
        refresh_per_second=5
    ) as progress:
        # Track start time and completed models for time estimation
        start_time = time()
        completed_times = []
        
        # Overall progress task
        overall_task = progress.add_task(
            "[bold cyan]Overall Progress", 
            total=len(config_manager.models)
        )
        
        # Task for current model's steps
        model_task = progress.add_task(
            "Current Model Steps",
            total=100,
            visible=False
        )
        
        results = []
        total_papers = len(papers)
        total_models = len(config_manager.models)
        
        for model_name, model_path in config_manager.models.items():
            model_start_time = time()
            try:
                # Calculate average time per model if we have data
                if completed_times:
                    avg_time = sum(completed_times) / len(completed_times)
                    remaining_models = total_models - len(completed_times)
                    est_total = avg_time * remaining_models
                    time_remaining = format_time(est_total)
                    progress.update(overall_task, 
                                  description=f"[bold cyan]Processing Models ({len(results)}/{total_models}) - Est. {time_remaining} remaining")
                else:
                    progress.update(overall_task, 
                                  description=f"[bold cyan]Processing Models ({len(results)}/{total_models})")
                
                # Reset and show model-specific task with model name always visible
                model_display_name = f"[bold yellow]{model_name}[/bold yellow]"
                progress.update(model_task, 
                              completed=0,
                              description=f"{model_display_name}: Preparing evaluation",
                              visible=True)
                
                scores = evaluator.evaluate_model(
                    papers=papers,
                    model_name=model_path,
                    progress=progress,
                    progress_task=model_task,
                    total_papers=total_papers
                )
                
                # Record time taken for this model
                model_time = time() - model_start_time
                completed_times.append(model_time)
                
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
                progress.advance(overall_task)
                console.print(f"✅ Processed: [green]{model_name}[/green]")

            except Exception as e:
                console.print(f"❌ Failed: [red]{model_name}[/red] - {str(e)}")
    
    # Save results
    console.print("\n[yellow]Saving results...[/yellow]")
    results_df = pd.DataFrame(results)
    results_df.to_csv(experiment_dir / 'embedding_comparison_results.csv', index=False)
    evaluator.create_leaderboard(results_df, experiment_dir)

cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
