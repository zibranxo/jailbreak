"""Export PyTorch DistilBERT model to ONNX format (Phase 7)."""

import os
import torch
import logging
from pathlib import Path

logger = logging.getLogger("onnx_export")
logging.basicConfig(level=logging.INFO)

def export_to_onnx(model_path: str = "data/models/safety_classifier"):
    """
    Export the HuggingFace PyTorch model to ONNX format for optimized inference.
    """
    logger.info(f"Attempting to load PyTorch model from {model_path}...")
    
    if not Path(model_path).exists():
        logger.warning(f"Model path {model_path} does not exist. Cannot export to ONNX.")
        logger.info("In a real environment, this script would load the PyTorch model and call torch.onnx.export().")
        return
        
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        dummy_input = tokenizer("This is a dummy input for ONNX tracing.", return_tensors="pt")
        input_ids = dummy_input["input_ids"]
        attention_mask = dummy_input["attention_mask"]
        
        onnx_path = os.path.join(model_path, "model.onnx")
        
        logger.info(f"Exporting model to {onnx_path}...")
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            onnx_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                          'attention_mask': {0: 'batch_size', 1: 'sequence'},
                          'logits': {0: 'batch_size'}}
        )
        logger.info("ONNX export successful.")
        
    except ImportError:
        logger.error("transformers or torch not installed. Cannot perform ONNX export.")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")

if __name__ == "__main__":
    export_to_onnx()
