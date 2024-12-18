import numpy as np
import pandas as pd
import sys
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from rich.progress import Progress
from rich.table import Table
from rich.panel import Panel
from utils import console
from models import ModelManager
from datetime import datetime
import yaml
import shutil
from pathlib import Path

class Evaluator:
    """Handles evaluation and scoring of embeddings."""
    
    def __init__(self, config, model_manager: ModelManager):
        """Initialize the evaluator."""
        self.config = config
        self.model_manager = model_manager

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
        
        progress.update(task_id, description=f"[cyan]Computing embeddings for {model_name}[/cyan]")

        for paper1 in papers:
            # Get embeddings (will use cache if available)
            title_emb = self.model_manager.get_embedding(
                paper1['title'], 
                model_name,
                self.config.get_cache_path(paper1['title'], model_name)
            )
            abstract1_emb = self.model_manager.get_embedding(
                paper1['abstract'], 
                model_name,
                self.config.get_cache_path(paper1['abstract'], model_name)
            )
            
            # Title to own abstract
            sim = cosine_similarity([title_emb], [abstract1_emb])[0][0]
            results['title_abstract_same'].append(sim)
            
            for paper2 in papers:
                if paper1 != paper2:
                    abstract2_emb = self.model_manager.get_embedding(
                        paper2['abstract'], 
                        model_name,
                        self.config.get_cache_path(paper2['abstract'], model_name)
                    )
                    
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

            progress.update(task_id, advance=len(papers))
        
        return {k: (float(np.mean(v)), float(np.std(v))) for k, v in results.items()}

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
        
        stats = {
            'total_papers': len(papers),
            'papers_per_field': {
                field: len([p for p in papers if p['query_field'] == field])
                for field in self.config.fields
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
- `embeddings/`: Cached embeddings (if included)

## Reproduction
To reproduce this experiment:
1. Use the same configuration from `experiment_config.yaml`
2. Ensure the same package versions are installed
3. Run with the same papers from `papers_full.csv`
"""
        
        with open(experiment_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        if include_embeddings and self.config.cache_dir.exists():
            embeddings_dir = experiment_dir / 'embeddings'
            embeddings_dir.mkdir(exist_ok=True)
            for cache_file in self.config.cache_dir.glob('*.pkl'):
                shutil.copy2(cache_file, embeddings_dir / cache_file.name)
        
        console.print(f"\n💾 Experiment metadata saved to: [cyan]{experiment_dir}[/cyan]")
