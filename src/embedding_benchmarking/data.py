import arxiv
import pandas as pd
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict
from rich.progress import Progress
from .utils import console
from .models import ModelManager

# Constants for retry logic
MAX_RETRIES = 5
BASE_DELAY = 15  # Base delay in seconds
MAX_JITTER = 15  # Maximum random jitter in seconds


class DataManager:
    """Handles paper fetching and data management."""
    
    def __init__(self, config, model_manager: ModelManager):
        """Initialize the data manager."""
        self.config = config
        self.model_manager = model_manager

    def fetch_papers_for_field(self,
                             query: str,
                             max_attempts: int = 1000) -> List[Dict]:
        """Fetch papers for a specific field with retry logic."""
        field_papers = []
        attempts = 0
        rejected = {'too_short': 0, 'too_long': 0}
        retry_count = 0

        while retry_count < MAX_RETRIES:
            try:
                # Add delay with jitter before each retry
                if retry_count > 0:
                    delay = BASE_DELAY * (2 ** (retry_count - 1)) + random.uniform(0, MAX_JITTER)
                    console.print(f"Retrying in {delay:.1f} seconds (attempt {retry_count + 1}/{MAX_RETRIES})...")
                    time.sleep(delay)

                client = arxiv.Client()
                search = arxiv.Search(
                    query=query,
                    max_results=max_attempts,
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )

                # Track if we've received any results
                received_results = False
                
                for result in client.results(search):
                    received_results = True
                    attempts += 1
                    meets_length, token_count = self.model_manager.check_text_length(
                        result.summary,
                        self.config.min_tokens,
                        self.config.max_tokens
                    )

                    # Add small delay between processing results to avoid overwhelming the API
                    #time.sleep(0.1)

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
                        
                        if len(field_papers) >= self.config.papers_per_field:
                            break
                    else:
                        if token_count < self.config.min_tokens:
                            rejected['too_short'] += 1
                        else:
                            rejected['too_long'] += 1
                    
                    if attempts >= max_attempts:
                        break
            
                # Check if we received any results
                if not received_results:
                    raise arxiv.ArxivError("No results received from arXiv API")

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

                # If we get here, we've successfully processed the results
                return field_papers[:self.config.papers_per_field]

            except arxiv.ArxivError as e:
                retry_count += 1
                console.print(f"⚠️  arXiv API error (attempt {retry_count}/{MAX_RETRIES}): {str(e)}")
                if retry_count >= MAX_RETRIES:
                    console.print(f"❌ Max retries exceeded for query: [red]{query}[/red]")
                    return []
                continue

            except Exception as e:
                console.print(f"❌ Unexpected error fetching papers for [red]{query}[/red]: {str(e)}")
                return []

    def fetch_papers(self) -> List[Dict]:
        """Fetch all papers."""
        papers = []
        papers_per_field = self.config.papers_per_field
        
        for query in self.config.fields:
            try:
                field_papers = self.fetch_papers_for_field(query)
                papers.extend(field_papers)
                
            except ValueError as e:
                console.print(f"\n⚠️ Warning for field '{query}': {str(e)}")
                return []
        
        self._print_collection_summary(papers)
        
        
        console.print(f"\n✅ Successfully collected {len(papers)} papers "
                     f"({papers_per_field} per field)")
        
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
