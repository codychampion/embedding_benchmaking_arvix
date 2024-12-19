import warnings
import logging
import os
from rich.console import Console

# Configure console for rich output
console = Console()

def setup_logging():
    """Configure logging and warnings."""
    warnings.filterwarnings('ignore')
    logging.getLogger("transformers").setLevel(logging.ERROR)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Suppress HuggingFace download messages
    for logger in ["huggingface_hub.file_download", 
                  "huggingface_hub.utils._validators",
                  "huggingface_hub.hub_mixin"]:
        logging.getLogger(logger).setLevel(logging.ERROR)
