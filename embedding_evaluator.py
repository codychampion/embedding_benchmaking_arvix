"""Academic Embedding Model Evaluator.

This module provides backward compatibility with the original implementation
while using the new modular structure internally.
"""

import os
from typing import Optional
from dotenv import load_dotenv

from .config import Config
from .models import ModelManager
from .data import DataManager
from .evaluation import Evaluator
from .utils import console, setup_logging
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

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
        setup_logging()
        
        # Initialize components
        self.config = Config(
            config_path=config_path,
            cache_dir=cache_dir,
            papers_per_field=papers_per_field,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            device=device
        )
        
        self.model_manager = ModelManager(
            device=device,
            hf_token=os.getenv('HUGGINGFACE_TOKEN')
        )
        
        self.data_manager = DataManager(self.config, self.model_manager)
        self.evaluator = Evaluator(self.config, self.model_manager)

    def run_comparison(self, output_dir: str = '.') -> None:
        """Run the complete comparison pipeline."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            # Fetch papers
            papers = self.data_manager.fetch_papers(progress)
            
            if not papers:
                console.print("\n❌ Paper collection failed. Aborting comparison.")
                return
                
            if len(papers) == len(self.config.fields) * self.config.papers_per_field:
                self.evaluator.save_experiment_metadata(papers, output_dir)
                
                results = []
                total_comparisons = len(papers) * len(self.config.models)
                eval_task = progress.add_task(
                    "[cyan]Evaluating models[/cyan]",
                    total=total_comparisons
                )
                
                for model_name, model_path in self.config.models.items():
                    try:
                        scores = self.evaluator.evaluate_model(papers, model_path, progress, eval_task)
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
                
                import pandas as pd
                results_df = pd.DataFrame(results)
                results_df.to_csv('embedding_comparison_results.csv', index=False)
                self.evaluator.create_leaderboard(results_df)
            else:
                console.print("\n❌ Incorrect number of papers collected. Aborting comparison.")

if __name__ == "__main__":
    # For backward compatibility with direct script execution
    load_dotenv()
    
    if not os.getenv('HUGGINGFACE_TOKEN'):
        console.print("⚠️  [red]HUGGINGFACE_TOKEN not found in environment variables![/red]")
        console.print("Please set your HuggingFace token in the .env file")
        exit(1)

    console.print("\n[bold green]🚀 Starting Academic Embedding Model Evaluation[/bold green]")
    
    evaluator = EmbeddingEvaluator(
        cache_dir='embedding_cache',
        papers_per_field=25,
        max_tokens=512,
        min_tokens=50,
        config_path='config.yaml' if os.path.exists('config.yaml') else None
    )
    evaluator.run_comparison()
