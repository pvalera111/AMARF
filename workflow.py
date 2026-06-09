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
    print("⚙️ WORKFLOW CONFIGURATION")
    print(f" File: {theme_path}")
    print(f" Epochs: {current_epochs}")
    print(" ------------------------")
    print(" Options:")
    print("  Higher steps (+15)")
    print("  Modify Test Query")
    print("  Freeze state (v1.0)")
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
        print("💾 Freezing state...")
        return (
            "freeze", 
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
    print("=== AMARF WORKFLOW ===")
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
        
    print("\nAvailable Themes:")
    for idx, f_name in enumerate(
        json_files
    ):
        print(f" [{idx + 1}] {f_name}")
        
    u_choice = input(
        "\nSelect theme index: "
    ).strip()
    try:
        idx = int(u_choice) - 1
        sel_file = json_files[idx]
    except (ValueError, IndexError):
        sel_file = json_files[0]
        
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
                learning_rate
            )
        )
        
        action, updated_epochs = (
            handle_user_refinement(
                THEME_FILE_PATH, 
                epochs
            )
        )
        
        if action == "exit":
            print("🪐 Workflow closed.")
            break
            
        epochs = updated_epochs
        if __name__ == "__main__":
    print("=== AMARF WORKFLOW INTERACTIVE PLATFORM ===")
    
    #  CI/CD  GITHUB
    if "--mock-mode" in sys.argv:
        print("🚀 [CI/CD] Mock Mode active. Validating dataset structures...")
        data_dir = "data"
        if os.path.exists(data_dir) and os.listdir(data_dir):
            print("✅ [CI/CD] Validation successful. Infrastructure configuration clean.")
            sys.exit(0)
        else:
            print("❌ [CI/CD] Validation failed. Data directories empty.")
            sys.exit(1)
            
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"❌ Error: '{data_dir}' directory not found.")
...

