import arxiv
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
from rich.progress import Progress
from utils import console
from models import ModelManager

class DataManager:
    """Handles paper fetching and data management."""
    
    def __init__(self, config, model_manager: ModelManager):
        """Initialize the data manager."""
        self.config = config
        self.model_manager = model_manager

    def fetch_papers_for_field(self, 
                             query: str, 
                             progress: Progress,
                             task_id: int,
                             max_attempts: int = 1000) -> List[Dict]:
        """Fetch papers for a specific field."""
        field_papers = []
        attempts = 0
        rejected = {'too_short': 0, 'too_long': 0}
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_attempts,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        try:
            for result in client.results(search):
                attempts += 1
                meets_length, token_count = self.model_manager.check_text_length(
                    result.summary,
                    self.config.min_tokens,
                    self.config.max_tokens
                )
                
                if meets_length:
                    paper = {
                        'title': result.title,
                        'abstract': result.summary,
                        'token_count': token_count,
                        'arxiv_id': result.entry_id.split('/')[-1],
                        'category': result.primary_category,
                        'categories': ','.join(result.categories),
                        'query_field': query,
                        'published_date': result.published.strftime('%Y-%m-%d'),
                        'collection_timestamp': datetime.now().isoformat(),
                    }
                    
                    field_papers.append(paper)
                    progress.update(task_id, advance=1)
                    
                    if len(field_papers) >= self.config.papers_per_field:
                        break
                else:
                    if token_count < self.config.min_tokens:
                        rejected['too_short'] += 1
                    else:
                        rejected['too_long'] += 1
                
                if attempts >= max_attempts:
                    break
            
            # Print statistics for this query
            console.print(f"\n📊 Query statistics for [cyan]{query}[/cyan]:")
            console.print(f"   - Papers found: {len(field_papers)}")
            console.print(f"   - Papers checked: {attempts}")
            console.print(f"   - Rejected (too short): {rejected['too_short']}")
            console.print(f"   - Rejected (too long): {rejected['too_long']}")
            
            if not field_papers:
                console.print(f"⚠️  No valid papers found for query: [red]{query}[/red]")
                console.print(f"   Current criteria: {self.config.min_tokens}-{self.config.max_tokens} tokens")
                return []
            
            if len(field_papers) < self.config.papers_per_field:
                console.print(f"⚠️  Only found {len(field_papers)} papers for query: [yellow]{query}[/yellow]")
                console.print(f"   Needed: {self.config.papers_per_field}")
                return []
            
            return field_papers[:self.config.papers_per_field]
            
        except Exception as e:
            console.print(f"❌ Error fetching papers for [red]{query}[/red]: {str(e)}")
            return []

    def fetch_papers(self, progress: Progress) -> List[Dict]:
        """Fetch all papers."""
        papers = []
        papers_task = progress.add_task(
            "Collecting papers...",
            total=len(self.config.fields) * self.config.papers_per_field
        )
        
        for query in self.config.fields:
            try:
                field_papers = self.fetch_papers_for_field(
                    query, 
                    progress,
                    papers_task
                )
                papers.extend(field_papers)
                
            except ValueError as e:
                console.print(f"\n⚠️ Warning for field '{query}': {str(e)}")
                return []
        
        self._print_collection_summary(papers)
        
        if len(papers) != len(self.config.fields) * self.config.papers_per_field:
            console.print(f"\n❌ Error: Got {len(papers)} papers, "
                        f"expected {len(self.config.fields) * self.config.papers_per_field}")
            return []
        
        console.print(f"\n✅ Successfully collected {len(papers)} papers "
                     f"({self.config.papers_per_field} per field)")
        
        return papers

    def _print_collection_summary(self, papers: List[Dict]) -> None:
        """Print summary of collected papers."""
        from rich.table import Table
        
        console.print("\n📊 Paper Collection Summary:")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Field")
        table.add_column("Papers")
        table.add_column("Avg Tokens")
        table.add_column("Token Range")
        
        for field in self.config.fields:
            field_papers = [p for p in papers if p['query_field'] == field]
            token_counts = [p['token_count'] for p in field_papers]
            
            if token_counts:  # Only add row if we have papers for this field
                table.add_row(
                    field,
                    str(len(field_papers)),
                    f"{sum(token_counts)/len(token_counts):.1f}",
                    f"{min(token_counts)}-{max(token_counts)}"
                )
        
        console.print(table)
