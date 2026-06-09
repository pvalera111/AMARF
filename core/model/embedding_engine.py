# core/model/embedding_engine.py
import os
import json
import numpy as np

class AMARFEmbeddingEngine:
    def __init__(self, model_name: str = "base_model"):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        local_model_path = os.path.join(base_dir, model_name)
        
        print(f"📦 [AMARF-MODEL] Initializing autonomous core...")
        print(f"📡 Mode: OFFLINE AIR-GAPPED. Local path:\n ↳ {local_model_path}")
        
        config_path = os.path.join(local_model_path, "config.json")
        vocab_path = os.path.join(local_model_path, "vocab.txt")

        fallback_config = {
            "architectures": ["DistilBertForMaskedLM"],
            "attention_dropout": 0.1, "dim": 768, "dropout": 0.1, "hidden_dim": 3072,
            "initializer_range": 0.02, "max_position_embeddings": 512, "model_type": "distilbert",
            "n_heads": 12, "n_layers": 6, "pad_token_id": 0, "qa_dropout": 0.1,
            "seq_classif_dropout": 0.2, "sinusoidal_pos_embds": False, "transformers_version": "4.36.0",
            "vocab_size": 30522
        }

        config_data = None
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception:
                pass
        
        if config_data is None:
            os.makedirs(local_model_path, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(fallback_config, f, indent=2)
            print("✅ File config.json restored successfully.")

        if not os.path.exists(vocab_path) or os.path.getsize(vocab_path) < 10:
            print("📝 Generating fallback autonomous vocabulary matrix (vocab.txt)...")
            tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            for char in "abcdefghijklmnopqrstuvwxyz0123456789-_.=:/?":
                tokens.append(char)
            keywords = ["rhel", "network", "config", "static", "ip", "eth0", "nmcli", "connection", 
                        "modify", "ipv4", "addresses", "manual", "security", "firewall", "cmd", 
                        "zone", "remove", "service", "permanent", "storage", "lsblk", "size"]
            tokens.extend(keywords)
            while len(tokens) < 30522:
                tokens.append(f"unused_{len(tokens)}")
                
            with open(vocab_path, "w", encoding="utf-8") as f:
                f.write("\n".join(tokens) + "\n")
            print("✅ Reference dictionary vocab.txt generated successfully.")

        try:
            import torch
            from transformers import DistilBertTokenizer, AutoModel
            self.framework = "pytorch"
            print("💎 PyTorch runtime environment detected. Launching inference.")
            
            self.tokenizer = DistilBertTokenizer(vocab_file=vocab_path, do_lower_case=True)
            self.model = AutoModel.from_pretrained(local_model_path, local_files_only=True)
            
        except (ImportError, AttributeError) as pt_err:
            try:
                import tensorflow as tf
                from transformers import DistilBertTokenizer, TFAutoModelForMaskedLM
                self.framework = "tensorflow"
                print("🔶 TensorFlow runtime environment detected. Launching inference.")
                
                self.tokenizer = DistilBertTokenizer(vocab_file=vocab_path, do_lower_case=True)
                self.model = TFAutoModelForMaskedLM.from_pretrained(local_model_path, local_files_only=True)
            except ImportError:
                print("❌ CRITICAL ERROR: Runtime ML frameworks not found.")
                raise pt_err

    def _mean_pooling_pt(self, model_output, attention_mask) -> np.ndarray:
        import torch
        if hasattr(model_output, "last_hidden_state"):
            token_embeddings = model_output.last_hidden_state
        else:
            token_embeddings = model_output if isinstance(model_output, tuple) else model_output
            
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return (sum_embeddings / sum_mask).detach().cpu().numpy()

    def _mean_pooling_tf(self, model_output, attention_mask) -> np.ndarray:
        import tensorflow as tf
        if hasattr(model_output, "last_hidden_state"):
            token_embeddings = model_output.last_hidden_state
        elif hasattr(model_output, "hidden_states") and model_output.hidden_states is not None:
            token_embeddings = model_output.hidden_states[-1]
        else:
            token_embeddings = model_output if isinstance(model_output, tuple) else model_output
            
        input_mask_expanded = tf.cast(tf.expand_dims(attention_mask, -1), tf.float32)
        sum_embeddings = tf.reduce_sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = tf.reduce_sum(input_mask_expanded, 1)
        sum_mask = tf.maximum(sum_mask, 1e-9)
        return (sum_embeddings / sum_mask).numpy()

    def get_vector(self, text: str) -> np.ndarray:
        vectors = self.get_vectors_batch([text])
        return vectors.flatten()

    def get_vectors_batch(self, texts: list[str]) -> np.ndarray:
        if self.framework == "pytorch":
            import torch
            inputs = self.tokenizer(texts, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            return self._mean_pooling_pt(outputs, inputs['attention_mask'])
        else:
            inputs = self.tokenizer(texts, max_length=64, padding="max_length", truncation=True, return_tensors="tf")
            outputs = self.model(inputs, output_hidden_states=True)
            return self._mean_pooling_tf(outputs, inputs['attention_mask'])
