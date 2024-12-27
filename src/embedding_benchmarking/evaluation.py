import numpy as np
import pandas as pd
import sys
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from rich.table import Table
from rich.panel import Panel
from .utils import console
from .models import ModelManager
from datetime import datetime
import yaml
from pathlib import Path

class Evaluator:
    """Handles evaluation and scoring of embeddings."""
    
    def __init__(self, config, model_manager: ModelManager):
        """Initialize the evaluator."""
        self.config = config
        self.model_manager = model_manager

    def _format_time(self, seconds: float) -> str:
        """Format time in appropriate units."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def evaluate_model(self,
                      papers: List[Dict],
                      model_name: str,
                      progress=None,
                      progress_task=None,
                      total_papers=None) -> Dict[str, Tuple[float, float]]:
        """Evaluate a model on papers using batched operations."""
        from time import time
        
        # Initialize timing dictionary and batch size
        stage_times = {}
        batch_size = min(512, total_papers)  # Smaller batches for more frequent updates
        
        results = {
            'title_abstract_same': [],
            'title_abstract_diff': [],
            'title_abstract_other': [],
            'abstract_abstract_same': [],
            'abstract_abstract_diff': []
        }
        
        # Get all titles and abstracts
        titles = [paper['title'] for paper in papers]
        abstracts = [paper['abstract'] for paper in papers]
        categories = [paper['category'] for paper in papers]
        
        # Start timing for title embeddings
        start_time = time()
        title_embeddings = []
        
        # Process titles in batches
        for i in range(0, len(titles), batch_size):
            batch = titles[i:i+batch_size]
            if progress and progress_task:
                progress.update(progress_task, 
                              description=f"[yellow]Getting title embeddings ({i+len(batch)}/{total_papers})[/yellow]",
                              completed=10 + (20 * (i+len(batch)) / len(titles)))
            batch_embeddings = self.model_manager.get_embeddings(batch, model_name)
            title_embeddings.extend(batch_embeddings)
        
        title_embeddings = np.array(title_embeddings)
        stage_times['title_embeddings'] = time() - start_time
        
        # Start timing for abstract embeddings
        start_time = time()
        abstract_embeddings = []
        
        # Process abstracts in batches
        for i in range(0, len(abstracts), batch_size):
            batch = abstracts[i:i+batch_size]
            if progress and progress_task:
                est_time = stage_times['title_embeddings'] * ((len(abstracts) - i) / len(titles))
                time_str = self._format_time(est_time)
                progress.update(progress_task, 
                              description=f"[yellow]Getting abstract embeddings ({i+len(batch)}/{total_papers}) - {time_str} remaining[/yellow]",
                              completed=30 + (20 * (i+len(batch)) / len(abstracts)))
            batch_embeddings = self.model_manager.get_embeddings(batch, model_name)
            abstract_embeddings.extend(batch_embeddings)
        
        abstract_embeddings = np.array(abstract_embeddings)
        stage_times['abstract_embeddings'] = time() - start_time
        
        # Start timing for similarity computations
        start_time = time()
        if progress and progress_task:
            est_time = (len(papers) ** 2) * 0.0001  # Rough estimate for similarity computation
            time_str = self._format_time(est_time)
            progress.update(progress_task,
                          description=f"[yellow]Computing title-abstract similarities - {time_str} estimated[/yellow]",
                          completed=50)
        
        # Title to own abstract similarities
        title_own_abstract_sims = np.diagonal(cosine_similarity(title_embeddings, abstract_embeddings))
        results['title_abstract_same'].extend(title_own_abstract_sims.tolist())
        stage_times['similarity_computation'] = time() - start_time
        
        # Start timing for pairwise computations
        start_time = time()
        if progress and progress_task:
            # Estimate time based on paper count (quadratic complexity)
            est_time = (len(papers) ** 2) * (stage_times['similarity_computation'] / len(papers))
            time_str = self._format_time(est_time)
            progress.update(progress_task,
                          description=f"[yellow]Computing pairwise similarities - {time_str} estimated[/yellow]",
                          completed=70)
        
        # All pairwise similarities
        title_abstract_sims = cosine_similarity(title_embeddings, abstract_embeddings)
        abstract_abstract_sims = cosine_similarity(abstract_embeddings, abstract_embeddings)
        stage_times['pairwise_computation'] = time() - start_time
        
        # Start timing for score processing
        start_time = time()
        if progress and progress_task:
            # Estimate remaining time based on paper count
            est_time = (len(papers) ** 2) * 0.0001  # Rough estimate for score processing
            progress.update(progress_task,
                          description=f"[yellow]Processing similarity scores (~{est_time:.1f}s)[/yellow]",
                          completed=90)
        
        # Process pairwise similarities
        n = len(papers)
        for i in range(n):
            for j in range(n):
                if i != j:  # Skip self-comparisons
                    if categories[i] == categories[j]:
                        results['title_abstract_diff'].append(title_abstract_sims[i, j])
                        results['abstract_abstract_same'].append(abstract_abstract_sims[i, j])
                    else:
                        results['title_abstract_other'].append(title_abstract_sims[i, j])
                        results['abstract_abstract_diff'].append(abstract_abstract_sims[i, j])
        stage_times['score_processing'] = time() - start_time
        
        # Calculate total processing time
        total_time = sum(stage_times.values())
        if progress and progress_task:
            time_str = self._format_time(total_time)
            progress.update(progress_task,
                          description=f"[green]Evaluation complete (Total: {time_str})[/green]",
                          completed=100)
        
        return {k: (float(np.mean(v)), float(np.std(v))) for k, v in results.items()}

    def create_leaderboard(self, results_df: pd.DataFrame, experiment_dir: Path) -> pd.DataFrame:
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
        
        # Save leaderboard to experiment directory
        leaderboard_path = experiment_dir / 'model_leaderboard.csv'
        leaderboard.to_csv(leaderboard_path, index=False)
        console.print(f"\n💾 Leaderboard saved to: [cyan]{leaderboard_path}[/cyan]")

        return leaderboard

    def save_experiment_metadata(self, 
                               papers: List[Dict], 
                               experiment_dir: Path) -> None:
        """Save experiment metadata."""
        # Create experiment directory if it doesn't exist
        experiment_dir.mkdir(parents=True, exist_ok=True)

        config = {
            'timestamp': datetime.now().isoformat(),
            'papers_per_field': self.config.papers_per_field,
            'max_tokens': self.config.max_tokens,
            'min_tokens': self.config.min_tokens,
            'fields': self.config.fields,
            'models': self.config.models,
            'device': self.config.device,
            'python_version': sys.version
        }
        
        with open(experiment_dir / 'experiment_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        paper_df = pd.DataFrame(papers)
        paper_df.to_csv(experiment_dir / 'papers_full.csv', index=False)
        clean_df = paper_df.drop(columns=['abstract'])
        clean_df.to_csv(experiment_dir / 'papers_metadata.csv', index=False)
        
        # Convert numpy values to Python native types
        token_stats = {
            'mean': float(paper_df['token_count'].mean()),
            'median': float(paper_df['token_count'].median()),
            'min': int(paper_df['token_count'].min()),
            'max': int(paper_df['token_count'].max()),
            'std': float(paper_df['token_count'].std())
        }
        
        # Convert string dates to datetime objects before using strftime
        paper_df['published_date'] = pd.to_datetime(paper_df['published_date'])
        
        stats = {
            'total_papers': len(papers),
            'papers_per_field': {
                field: len([p for p in papers if p['query_field'] == field])
                for field in self.config.fields
            },
            'token_statistics': token_stats,
            'date_range': {
                'earliest': paper_df['published_date'].min().strftime('%Y-%m-%d'),
                'latest': paper_df['published_date'].max().strftime('%Y-%m-%d')
            }
        }
        
        with open(experiment_dir / 'collection_statistics.yaml', 'w') as f:
            yaml.dump(stats, f, default_flow_style=False)
        
        readme_content = f"""# Embedding Evaluation Experiment

## Experiment Details
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Papers per field: {self.config.papers_per_field}
- Total papers: {len(papers)}
- Token range: {self.config.min_tokens}-{self.config.max_tokens}

## Fields
{chr(10).join(f'- {field}' for field in self.config.fields)}

## Models
{chr(10).join(f'- {model}' for model in self.config.models.keys())}

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
"""
        
        with open(experiment_dir / 'README.md', 'w') as f:
            f.write(readme_content)
