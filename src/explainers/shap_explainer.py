"""SHAP-based explainer for the ML text classifier."""

import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SHAPExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Use a text explainer
        # Wrapping model in a pipeline or custom function is typically required
        # for Hugging Face models in SHAP.
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
        self.explainer = shap.Explainer(self.pipe)

    def explain(self, text: str, target_class: str = "jailbreak") -> list[tuple[str, float]]:
        """Run SHAP text explainer and return top contributing tokens."""
        try:
            shap_values = self.explainer([text])
            # shap_values is an Explanation object
            # shape: (1, num_tokens, num_classes)
            
            # Find the index of the target class
            class_idx = -1
            for i, label_dict in enumerate(self.pipe.model.config.id2label.values()):
                if label_dict == target_class or str(label_dict).lower() == target_class.lower():
                    class_idx = i
                    break
            
            # If we couldn't find it, just use the last class (which is often jailbreak)
            if class_idx == -1:
                class_idx = 2 # Assuming safe=0, toxic=1, jailbreak=2

            tokens = shap_values.data[0]
            values = shap_values.values[0, :, class_idx]
            
            # Filter out subword tokens and pair them
            contributions = [(str(t), float(v)) for t, v in zip(tokens, values)]
            
            # Sort by absolute contribution to highlight the most important tokens
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return contributions[:5]  # return top 5
        except Exception as e:
            import logging
            logging.getLogger("shap_explainer").warning(f"SHAP explanation failed: {e}")
            return []
