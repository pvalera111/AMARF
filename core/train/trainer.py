# core/train/trainer.py
import os
import time
import sqlite3
import torch
import torch.nn.functional as F
from torch.utils.data import (
    Dataset, DataLoader
)
from core.train.loss_functions import (
    AMARFContrastiveLoss
)

class AMARFTrainingDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx]

class AMARFTrainingEngine:
    def __init__(self, embedding_engine):
        self.ai = embedding_engine
        self.device = torch.device(
            "cuda" if 
            torch.cuda.is_available() 
            else "cpu"
        )
        print(f"🧬 [AMARF-TRAIN] Core "
              f"ready on: {self.device}")

    def _tensor_mean_pooling(self, 
                             out, 
                             mask):
        # ТОЧНИЙ ФІКС АТРИБУТУ SIZE
        if hasattr(
            out, "last_hidden_state"
        ):
            emb = out.last_hidden_state
        else:
            emb = out[0]

        mask_exp = (
            mask.unsqueeze(-1)
            .expand(emb.size())
            .float()
        )
        sum_emb = torch.sum(
            emb * mask_exp, 1
        )
        sum_mask = torch.clamp(
            mask_exp.sum(1), 
            min=1e-9
        )
        return sum_emb / sum_mask

    def train_agent_space(
        self, agent_id: int, 
        db_path: str = "storage.db", 
        epochs: int = 20, 
        lr: float = 1e-4
    ):
        print(f"\n🚀 Optimization "
              f"loop for ID: {agent_id}")
        
        pairs = []
        with sqlite3.connect(
            db_path
        ) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT text_content "
                "FROM knowledge_chunks "
                "WHERE agent_id = ?", 
                (agent_id,)
            )
            rows = cursor.fetchall()
            
        if not rows:
            print("⚠️ No items.")
            return

        for row in rows:
            text = row[0]
            pairs.append((text, text))

        model = self.ai.model.to(
            self.device
        )
        tokenizer = self.ai.tokenizer
        model.train()
        
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr
        )
        loss_fn = (
            AMARFContrastiveLoss(
                scale=20.0
            )
        )

        dataset = AMARFTrainingDataset(
            pairs
        )
        loader = DataLoader(
            dataset, 
            batch_size=len(pairs), 
            shuffle=False
        )

        for epoch in range(epochs):
            t_start = time.perf_counter()
            total_loss = 0.0
            for q, p in loader:
                optimizer.zero_grad()
                
                q_in = tokenizer(
                    list(q), max_length=64, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                ).to(self.device)
                
                p_in = tokenizer(
                    list(p), max_length=64, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                ).to(self.device)
                
                q_out = model(**q_in)
                p_out = model(**p_in)
                
                q_tensor = (
                    self
                    ._tensor_mean_pooling(
                        q_out, 
                        q_in['attention_mask']
                    )
                )
                p_tensor = (
                    self
                    ._tensor_mean_pooling(
                        p_out, 
                        p_in['attention_mask']
                    )
                )
                
                loss = loss_fn(
                    q_tensor, p_tensor
                )
                loss.backward()
                
                torch.nn.utils\
                .clip_grad_norm_(
                    model.parameters(), 
                    max_norm=1.0
                )
                optimizer.step()
                total_loss += loss.item()
                
            t_elapsed = (
                time.perf_counter() - 
                t_start
            ) * 1000
                
            if (
                (epoch + 1) % 5 == 0 
                or epoch == 0
            ):
                print(f"  📅 Step "
                      f"{epoch+1:02d}/"
                      f"{epochs:02d} | "
                      f"📉 Loss: "
                      f"{total_loss:.4f}")
                
                try:
                    from workflow import (
                        AMARFTelemetryEngine
                    )
                    tel = (
                        AMARFTelemetryEngine()
                    )
                    tel.log_event(
                        module_name="ANALYTICS",
                        status_code="METRIC",
                        execution_time_ms=
                        t_elapsed,
                        payload={
                            "epoch": 
                            epoch + 1,
                            "loss": 
                            float(total_loss)
                        }
                    )
                except ImportError:
                    pass

        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(
                        __file__
                    )
                )
            )
        )
        save_path = os.path.join(
            base_dir, "output", 
            f"agent_{agent_id}_refined"
        )
        os.makedirs(
            save_path, exist_ok=True
        )
        model.save_pretrained(
            save_path
        )
        print(f"💾 [AMARF-TRAIN] Saved "
              f"to:\n ↳ {save_path}")
        model.eval()
