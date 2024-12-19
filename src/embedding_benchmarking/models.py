import torch
import numpy as np
import json
import warnings
import boto3
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Any, Tuple, List
import os

class ModelManager:
    """Handles model loading and embedding generation."""
    
    def __init__(self, device: str = None, hf_token: str = None):
        """Initialize the model manager."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token
        
        # Initialize model and tokenizer caches
        self._model_cache = {}
        self._tokenizer_cache = {}
        
        # Initialize length checking tokenizer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.length_tokenizer = AutoTokenizer.from_pretrained(
                'bert-base-uncased',
                token=self.hf_token
            )
            
        # Initialize Bedrock client
        try:
            self.bedrock = boto3.client(
                service_name='bedrock-runtime',
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
        except Exception as e:
            print(f"Warning: Failed to initialize Bedrock client: {str(e)}")
            self.bedrock = None

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

    def get_embeddings_batch(self, texts: List[str], model_name: str) -> np.ndarray:
        """Get embeddings for a batch of texts."""
        if model_name == 'Bedrock':
            return self._get_bedrock_embeddings_batch(texts)
        else:
            return self._get_hf_embeddings_batch(texts, model_name)

    def get_embedding(self, text: str, model_name: str) -> np.ndarray:
        """Get single embedding (uses batch processing internally)."""
        return self.get_embeddings_batch([text], model_name)[0]

    def _get_bedrock_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings from Bedrock for a batch of texts."""
        if self.bedrock is None:
            raise Exception("Bedrock client not initialized. Check AWS credentials and region.")
        
        embeddings = []
        for text in texts:  # Bedrock doesn't support true batching, so process sequentially
            try:
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
                embeddings.append(np.array(json.loads(response['body'].read())['embedding']))
            except Exception as e:
                raise Exception(f"Bedrock API error: {str(e)}")
        
        return embeddings

    def _get_hf_embeddings_batch(self, texts: List[str], model_name: str) -> np.ndarray:
        """Get embeddings from HuggingFace model for a batch of texts."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            model, tokenizer = self._get_model_and_tokenizer(model_name)
            
            # Process in batches of 32 to avoid memory issues
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = tokenizer(
                    batch_texts,
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
                
                # Normalize embeddings
                embeddings = embeddings.cpu().numpy()
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-9)
                
                all_embeddings.append(embeddings)
            
            return np.vstack(all_embeddings)
