# workflow.py
import os
import sys
import json
from core.control import (
    run_workflow_iteration
)

def handle_user_refinement(
    theme_path, current_epochs
):
    print("\n⚙️ WORKFLOW CONFIGURATION")
    print(f" File: {theme_path}")
    print(f" Epochs: {current_epochs}")
    print(" ------------------------")
    print(" Options:")
    print("  Higher steps (+15)")
    print("  Modify Test Query")
    print("  Ingest New Source")
    print("  Exit platform")
    
    choice = input(
        "\nEnter choice [1-4]: "
    ).strip()
    
    if choice == "1":
        return (
            "epochs", 
            current_epochs + 15
        )
    elif choice == "2":
        new_q = input(
            "Enter query: "
        ).strip()
        if new_q:
            with open(
                theme_path, "r", 
                encoding="utf-8"
            ) as f:
                data = json.load(f)
            data["test_suite"][
                "user_query"
            ] = new_q
            with open(
                theme_path, "w", 
                encoding="utf-8"
            ) as f:
                json.dump(
                    data, f, indent=2
                )
        return (
            "query_updated", 
            current_epochs
        )
    elif choice == "3":
        print("\n📂 AMARF-KNOWLEDGE: "
              "SOURCE INGESTION")
        src = input(
            "Paste URL/File path: "
        ).strip()
        rule = input(
            "Enter playbook rule: "
        ).strip()
        if src and rule:
            with open(
                theme_path, "r", 
                encoding="utf-8"
            ) as f:
                data = json.load(f)
            data["knowledge_chunks"]\
            .append({
                "content": f"Source: "
                f"{src} | Policy: {rule}",
                "syntax_type": 
                "documentation_text"
            })
            with open(
                theme_path, "w", 
                encoding="utf-8"
            ) as f:
                json.dump(
                    data, f, indent=2
                )
            print("✅ Source synchronized.")
        return (
            "source_appended", 
            current_epochs
        )
    elif choice == "4" or (
        choice.lower() == "exit"
    ):
        return (
            "exit", 
            current_epochs
        )
    return (
        "continue", 
        current_epochs
    )

if __name__ == "__main__":
    if "--mock-mode" in sys.argv:
        print("🚀 [CI/CD] Mock Mode active.")
        data_dir = "data"
        if os.path.exists(
            data_dir
        ) and os.listdir(data_dir):
            print("✅ Clean config.")
            sys.exit(0)
        else:
            print("❌ Empty config.")
            sys.exit(1)

    print("=== AMARF WORKFLOW INTERACTIVE TESTBENCH ===")
    
    # 🪐 УЛЬТИМАТИВНИЙ КОНФІГУРАТОР АРХІТЕКТУРИ AMARF-MODEL
    print("\n🤖 [AMARF-MODEL] Select Neural Core Core Architecture Class:")
    print("=====================================================================")
    print(" CLASS I: LARGE LANGUAGE MODELS (Core Reasoning & Tools)")
    print("     ↳ Option A: Llama-3.1-8B-Instruct (Meta) [~16GB FP16 | 128K Context]")
    print("       Task: Chain-of-Thought planning, autonomous local Tool Calling.")
    print("     ↳ Option B: Qwen2.5-Coder-7B-Instruct (Alibaba) [~15GB FP16]")
    print("       Task: Local code audits, SQL generation, DB schema automation.")
    print("---------------------------------------------------------------------")
    print(" CLASS II: VISION-LANGUAGE MODELS (Graphical/Vision Agents)")
    print("     ↳ Model: Qwen2.5-VL-7B-Instruct (Alibaba) [~15.5GB FP16]")
    print("       Train: Frozen Vision Tower + LoRA on LLM / Unfrozen Projector.")
    print("       Task: Parsing complex blueprints, data charts, OCR, RPA automation.")
    print("---------------------------------------------------------------------")
    print(" CLASS III: EMBEDDING MODELS (RAG & Knowledge Retrieval Memory)")
    print("     ↳ Model: BGE-M3 Multilingual (BAAI) [567M parameters | ~2.2GB Size]")
    print("       Train: Full Fine-Tuning (SentenceTransformers / Contrastive Learning).")
    print("       Task: Sovereign high-precision vector search (UA/EN), internal indices.")
    print("---------------------------------------------------------------------")
    print(" CLASS IV: AUDIO & TRANSCRIPTION MODELS (Voice Agents & Monitoring)")
    print("     ↳ Model: OpenAI Whisper Large v3 [1.54B parameters | ~3.1GB Size]")
    print("       Train: PEFT LoRA on q_proj and v_proj layers inside Seq2SeqTrainer.")
    print("       Task: Audio streaming audits, dictation, processing radio jargon.")
    print("---------------------------------------------------------------------")
    print(" CLASS V: EMBEDDED LIGHT INFRASTRUCTURE (Local Test Sandbox)")
    print("     ↳ Model: DistilBert Embedded Core [66M parameters | ~260MB Size]")
    print("       Task: Ultra-fast local intent mapping, CPU/IoT micro-hardware execution.")
    print("=====================================================================")
    
    m_opt = input("Select model target configuration [1-5]: ").strip()
    if m_opt == "1":
        sub_opt = input("  -> Enter sub-choice [A/B]: ").strip().upper()
        m_type = "Llama-3.1-8B-Instruct" if sub_opt == "A" else "Qwen2.5-Coder-7B"
    elif m_opt == "2":
        m_type = "Qwen2.5-VL-7B-Vision"
    elif m_opt == "3":
        m_type = "BGE-M3-Multilingual"
    elif m_opt == "4":
        m_type = "Whisper-v3-Audio"
    else:
        m_type = "Embedded-DistilBert"

    # 🪐 КОНФІГУРАТОР РЕЖИМІВ AMARF-CONTROL
    print("\n... [AMARF-CONTROL] Select Application Target UI Mode:")
    print("  Passive RAG Advisory Bot Mode -> UI recommendations & assistant charts")
    print("  Active Finite State Machine Sandbox (FSM) -> Isolated automated actions")
    c_opt = input("Select application mode [1-2]: ").strip()
    c_type = (
        "Passive-Advisory-Bot" 
        if c_opt == "1" 
        else "Active-FSM-Sandbox"
    )

    data_dir = "data"
    if not os.path.exists(data_dir):
        print("❌ Error: 'data' missing")
        sys.exit(1)
        
    json_files = [
        f for f in os.listdir(data_dir) 
        if f.endswith(".json")
    ]
    if not json_files:
        print("❌ Error: No JSON files")
        sys.exit(1)
        
    print("\nAvailable Theme Configurations:")
    for idx, f_name in enumerate(
        json_files
    ):
        print(f" [{idx + 1}] {f_name}")
        
    u_choice = input(
        "\nSelect theme index to initialize workspace: "
    ).strip()
    try:
        idx = int(u_choice) - 1
        sel_file = json_files[idx]
    except (ValueError, IndexError):
        sel_file = json_files
        
    THEME_FILE_PATH = os.path.join(
        data_dir, sel_file
    )
    epochs = 20
    learning_rate = 1e-4

    while True:
        last_score = (
            run_workflow_iteration(
                THEME_FILE_PATH, 
                epochs, 
                learning_rate,
                m_type,
                c_type
            )
        )
        action, updated_epochs = (
            handle_user_refinement(
                THEME_FILE_PATH, 
                epochs
            )
        )
        
        if action == "exit":
            print("🪐 Workflow closed safely. Ready for production.")
            break
            
        epochs = updated_epochs
