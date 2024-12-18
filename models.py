import torch
import numpy as np
import pickle
import json
import warnings
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Any, Tuple
import os
from pathlib import Path

class ModelManager:
    """Handles model loading and embedding generation."""
    
    def __init__(self, device: str = None, hf_token: str = None):
        """Initialize the model manager."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token
        
        # Initialize caches
        self._model_cache = {}
        self._tokenizer_cache = {}
        
        # Initialize length checking tokenizer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.length_tokenizer = AutoTokenizer.from_pretrained(
                'bert-base-uncased',
                token=self.hf_token
            )

    def _get_model_and_tokenizer(self, model_name: str) -> Tuple[Any, Any]:
        """Get or load model and tokenizer with caching."""
        if model_name not in self._model_cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    token=self.hf_token
                )
                model = AutoModel.from_pretrained(
                    model_name,
                    token=self.hf_token
                ).to(self.device)
                model.eval()

            self._model_cache[model_name] = model
            self._tokenizer_cache[model_name] = tokenizer

        return self._model_cache[model_name], self._tokenizer_cache[model_name]

    def check_text_length(self, text: str, min_tokens: int, max_tokens: int) -> Tuple[bool, int]:
        """Check if text meets length requirements."""
        tokens = self.length_tokenizer(text, truncation=False)['input_ids']
        token_count = len(tokens)
        return (min_tokens <= token_count <= max_tokens, token_count)

    def get_embedding(self, text: str, model_name: str, cache_path: Path) -> np.ndarray:
        """Get embedding with caching."""
        # Check cache first
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # Generate embedding if not cached
        try:
            if model_name == 'Bedrock':
                embedding = self._get_bedrock_embedding(text)
            else:
                embedding = self._get_hf_embedding(text, model_name)
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")

        # Cache the result
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)

        return embedding

    def _get_bedrock_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Bedrock."""
        response = self.bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="*/*",
            body=json.dumps({
                "inputText": text,
                "dimensions": 512,
                "normalize": True
            })
        )
        return np.array(json.loads(response['body'].read())['embedding'])

    def _get_hf_embedding(self, text: str, model_name: str) -> np.ndarray:
        """Get embedding from HuggingFace model."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            model, tokenizer = self._get_model_and_tokenizer(model_name)

                
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)

            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Normalize embedding
            embedding = embeddings[0].cpu().numpy()
            return embedding / np.linalg.norm(embedding)
