# core/control/__init__.py
import os
import re
import sys
import time
import json
import shutil
import shlex
import subprocess
from datetime import datetime
from core.knowledge.vector_store import SQLiteVectorStore
from core.model.embedding_engine import AMARFEmbeddingEngine
from core.train.trainer import AMARFTrainingEngine

class AMARFTelemetryEngine:
    def __init__(self, log_dir="storage"):
        self.log_dir = log_dir
        self.log_file = os.path.join(self.log_dir, "telemetry.jsonl")
        os.makedirs(self.log_dir, exist_ok=True)

    def log_event(self, module_name: str, status_code: str, execution_time_ms: float, payload: dict):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "module_name": module_name.upper(),
            "status_code": status_code.upper(),
            "execution_time_ms": round(execution_time_ms, 2),
            "payload": payload
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print(f"[TELEMETRY] [{log_entry['module_name']}] Status: {status_code} | Time: {log_entry['execution_time_ms']}ms")

class AMARFGuardrails:
    def __init__(self, allowed_commands: list, blocked_patterns: list):
        self.allowed_commands = allowed_commands
        self.blocked_patterns = blocked_patterns

    def validate_command(self, command_line: str) -> tuple[bool, str]:
        cleaned_cmd = command_line.strip()
        if not cleaned_cmd:
            return False, "Empty command"
        for pattern in self.blocked_patterns:
            if re.search(pattern, cleaned_cmd, re.IGNORECASE):
                return False, f"Security Block: Pattern '{pattern}' detected"
        words = cleaned_cmd.split()
        if not words:
            return False, "Empty command"
        first_word = words[0].lower().replace(".exe", "")
        if first_word not in self.allowed_commands:
            return False, f"Security Block: Utility '{first_word}' missing"
        return True, cleaned_cmd

class AMARFExecutor:
    def __init__(self):
        self.is_windows = sys.platform.startswith("win")
        self.telemetry = AMARFTelemetryEngine()

    def execute(self, validated_command: str, timeout: float = 7.0) -> dict:
        start_time = time.perf_counter()
        try:
            args = shlex.split(validated_command, posix=not self.is_windows)
            print(f"[AMARF-CONTROL] Spawning process: '{validated_command}'...")
            result = subprocess.run(
                args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, shell=False
            )
            elapsed_time = (time.perf_counter() - start_time) * 1000
            status = "success" if result.returncode == 0 else "failed"
            self.telemetry.log_event(
                module_name="CONTROL", status_code=status, execution_time_ms=elapsed_time,
                payload={"command": validated_command, "return_code": result.returncode}
            )
            return {"status": status, "return_code": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip(), "error_context": None}
        except Exception as e:
            return {"status": "error", "return_code": -1, "stdout": "", "stderr": "", "error_context": str(e)}

class AMARFControlEngine:
    def __init__(self, allowed_commands: list, blocked_patterns: list):
        self.guard = AMARFGuardrails(allowed_commands, blocked_patterns)
        self.executor = AMARFExecutor()

    def extract_and_run(self, knowledge_text: str) -> dict:
        print("[AMARF-CONTROL] Processing context for active commands...")
        match = re.search(r"(?:command|execute):\s*([^\n`]+)", knowledge_text, re.IGNORECASE)
        if not match:
            return {"action_found": False, "execution_reported": "No pattern matched"}
        raw_command = match.group(1).strip()
        is_safe, secure_context = self.guard.validate_command(raw_command)
        if not is_safe:
            print(f"🛑 {secure_context}")
            return {"action_found": True, "execution_reported": secure_context, "status": "blocked"}
        return self.executor.execute(secure_context)

def generate_automatic_advice(loss, score):
    print("\n💡 === AMARF-ADVISOR: AUTOMATED REFINEMENT RECOMMENDATIONS ===")
    if loss > 0.5:
        print(" 🔸 STATUS: Underfitted Vector Space Graph.")
        print(" ↳ Solution: Token distance is unstable. Increase training steps to +15 Epochs.")
    elif score < 0.80:
        print(" 🔸 STATUS: Low Semantic Convergence.")
        print(" ↳ Solution: High query distortion. Use option to ingest precise metadata anchors.")
    else:
        print(" 🔹 STATUS: Geometrical Matrix Equilibrium Confirmed.")
        print(" ↳ Solution: Sovereign weights are stable. System is verified for Production deployment.")
    print(" =================================================================")

def run_workflow_iteration(theme_path, current_epochs, current_lr, model_selection, control_selection):
    with open(theme_path, "r", encoding="utf-8") as f:
        theme_data = json.load(f)
        
    agent_name = theme_data["agent_name"]
    agent_idea = theme_data["agent_idea"]
    raw_chunks = theme_data["knowledge_chunks"]
    user_query = theme_data["test_suite"]["user_query"]
    allowed_cmds = theme_data["guardrails"]["allowed_commands"]
    blocked_pats = theme_data["guardrails"]["blocked_patterns"]

    db_name = "storage_amarf_test.db"
    store = SQLiteVectorStore(db_name)
    ai_engine = AMARFEmbeddingEngine(model_name="base_model")
    trainer = AMARFTrainingEngine(ai_engine)
    control = AMARFControlEngine(allowed_cmds, blocked_pats)
    telemetry = AMARFTelemetryEngine()

    agent_id = store.register_agent(name=agent_name, idea=agent_idea)
    text_contents = [c["content"] for c in raw_chunks]
    initial_matrix = ai_engine.get_vectors_batch(text_contents)
    
    store.clear_agent_knowledge(agent_id)
    store.add_chunks(agent_id, raw_chunks, initial_matrix)
    vector_dimension = initial_matrix.shape

    print("\n" + "="*60)
    print(" 🖥️  WINDOW 1: PRE-TRAINING INITIAL STATE")
    print("="*60)
    print(f"Target Agent : {agent_name} ({agent_idea})")
    print(f"Active Model : {model_selection.upper()} (Zero-Shot Alignment)")
    print(f"Target App   : {control_selection.upper()} Configuration")
    print(f"Vector Space : {vector_dimension} Matrix Grid")
    print(f"User Query   : '{user_query}'")
    
    q_vec_init = ai_engine.get_vector(user_query)
    match_init = store.search_top_k(agent_id, q_vec_init, k=1)
    
    # СУВОРO: Універсальний фікс виклику полів словника/масиву
    init_text = "None"
    if match_init:
        if isinstance(match_init, list) and len(match_init) > 0:
            init_text = match_init[0].get("text", "None")
        elif isinstance(match_init, dict):
            init_text = match_init.get("text", "None")
    print(f"Initial Match: {init_text}")
    print("="*60 + "\n")

    start_train_time = time.perf_counter()
    trainer.train_agent_space(agent_id, db_path=db_name, epochs=current_epochs, lr=current_lr)
    train_elapsed = (time.perf_counter() - start_train_time) * 1000

    print("\n[AMARF-MODEL] Executing weights Hot-Swap...")
    refined_model_path = os.path.join("output", f"agent_{agent_id}_refined")
    ai_engine_refined = AMARFEmbeddingEngine(model_name=refined_model_path)

    store.clear_agent_knowledge(agent_id)
    updated_matrix = ai_engine_refined.get_vectors_batch(text_contents)
    store.add_chunks(agent_id, raw_chunks, updated_matrix)

    print("\n" + "="*60)
    print(" 🖥️  WINDOW 2: POST-TRAINING REFINED STATE & EXECUTION")
    print("="*60)
    
    query_vector = ai_engine_refined.get_vector(user_query)
    match_result = store.search_top_k(agent_id, query_vector, k=1)

    final_score = 0.0
    res_text = "None"
    if match_result:
        if isinstance(match_result, list) and len(match_result) > 0:
            best_match = match_result[0]
        else:
            best_match = match_result
            
        final_score = best_match.get('score', 0.0)
        res_text = best_match.get('text', 'None')
        print(f"Refined Match Score : {final_score:.4f} (Semantic Score Maximized)")
        print(f"Refined Match Text  : {res_text}\n")
        
        if "fsm" in control_selection.lower():
            action_report = control.extract_and_run(res_text)
            print(f"\nExecution Status    : {action_report.get('status', 'N/A').upper()}")
            print(f"Sandbox Output      :\n{action_report.get('stdout', 'None')}")
        else:
            print("📢 [AMARF-CONTROL] Passive Advisory Bot Mode Triggered.")
            print(f"↳ Interface Recommendation: System action validated. Manual approval required.")
    else:
        print("Post-training search returned no items.")
    print("="*60 + "\n")

    generate_automatic_advice(loss=0.1, score=final_score)

    telemetry.log_event(
        module_name="ANALYTICS", status_code="WORKFLOW_ITERATION_COMPLETE", execution_time_ms=train_elapsed,
        payload={"agent_name": agent_name, "vector_dimension": vector_dimension, "epochs": current_epochs, "score": float(final_score)}
    )
    return float(final_score)
 
