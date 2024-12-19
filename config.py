import yaml
from pathlib import Path
from typing import Dict, List, Optional
from utils import console

class Config:
    """Handles configuration loading and default settings."""
    
    DEFAULT_MODELS = {
        'allenai/specter': 'allenai/specter',
        'allenai/scibert_scivocab_uncased': 'allenai/scibert_scivocab_uncased',
        'sentence-transformers/all-mpnet-base-v2': 'sentence-transformers/all-mpnet-base-v2',
    }
    
    DEFAULT_FIELDS = [
        "artificial intelligence",
        "machine learning",
        "computer vision",
    ]
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 cache_dir: str = 'embedding_cache',
                 papers_per_field: int = 25,
                 max_tokens: int = 512,
                 min_tokens: int = 50,
                 device: Optional[str] = None):
        """Initialize configuration."""
        self.cache_dir = Path(cache_dir)
        self.experiment_dir = None
        self.papers_per_field = papers_per_field
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.device = device
        
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
            self.models = self.DEFAULT_MODELS
            self.fields = self.DEFAULT_FIELDS
            console.print("ℹ️ Using default configuration")

    def set_experiment_dir(self, experiment_dir: str) -> None:
        """Set the experiment directory for caching embeddings."""
        self.experiment_dir = Path(experiment_dir)
        self.cache_dir = self.experiment_dir / 'embeddings'
        self.cache_dir.mkdir(exist_ok=True)

    def get_cache_path(self, text: str, model_name: str) -> Path:
        """Generate cache file path for a specific text and model."""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Replace all forward slashes with underscores in the model name
        model_name_safe = model_name.replace('/', '_')
        return self.cache_dir / f"{model_name_safe}_{text_hash}.pkl"
