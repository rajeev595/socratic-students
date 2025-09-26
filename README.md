# 🧑‍🏫🤖🧑‍🎓 Interact+ Experiments

This repository contains pipelines for curating and generating **in-distribution (ID)** and **out-of-distribution (OOD)** datasets for code and math reasoning tasks.  
The curated datasets are used to study **student–teacher interactions** with LLMs, track pass@k performance, and analyze difficulty.

---

## 📂 Directory Structure
```bash
codes/
├─ data/
│  ├─ code/
│  │  ├─ generate_ood.py      # Generate OOD code problems via teacher prompting
│  │  └─ sample_hard.py       # Curate hard in-distribution code problems
│  ├─ math/
│  │  ├─ generate_ood.py      # Build OOD math split (e.g., GSM8K+ alt variants)
│  │  └─ sample_hard.py       # Curate hard in-distribution math problems
│  ├─ rlhf.py                 # Collected SFT, DPO data for self-guided learning
│  └─ rlhf_guide.py           # Collected SFT, DPO data for peer-guided learning
│
├─ evaluation/
│  ├─ code.py                 # Code generation/evaluation utilities
│  ├─ math.py                 # Math generation/evaluation utilities
│  └─ utils.py                # Shared helpers (e.g., pass@k)
├─ prompts/
│  ├─ code.py                 # Prompt templates for code tasks
│  └─ math.py                 # Prompt templates for math tasks
├─ app.py                     # Streamlit dashboard for results/transcripts
├─ llm_utils.py               # OpenAI client builder, chat wrappers
├─ run_interact.py            # Student–teacher interaction loop
└─ run_sft_dpo.py             # SFT-DPO training script
└─ configs/
    └─ config.yaml                # Hydra config: paths, models, generation settings
```

---

## ⚙️ Installation

1. **Clone the repo** and enter the project root:
   ```bash
   git clone https://github.com/rajeev595/interact_plus.git
   cd interact_plus
   ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🛠️ Configuration

All scripts are configured through Hydra configs (configs/config.yaml).
Important keys include:

- `paths.data` → root data folder where curated sets are stored
- `paths.openai_api_key_fpath` → text file with your OpenAI API key
- `teacher.model`, `teacher.temperature` → teacher model settings
- `student.model`, `student.temperature` → student model settings
- `gen.k_values`, `gen.num_samples`, `gen.pass_k`, `gen.pass_at_k_threshold`
- `data.data_amount`, `data.opc_select_n`, `data.opc_seed`

You can override parameters via command line, e.g.:
```bash
python -m codes.data.code.sample_hard student.model="qwen7" data.data_amount=50
```

---

## 🚀 Usage

1. Curate hard problems (in-distribution)
    - Code
        ```bash
        python -m codes.data.code.sample_hard
        ```
    - Math (GSM-style)
        ```bash
        python -m codes.data.math.sample_hard
        ```

    These scripts:
    - Normalize datasets (HumanEval / GSM8K+)
    - Probe the student model → skip if too easy
    - Verify with the teacher model → keep only solvable problems
    - Save curated problems under:
        ```bash
        <data>/<teacher.model>/<dataset>/<student.model>.json
        ```

2. Generate OOD problems
    - Code (prompt teacher to create OOD problems)
        ```bash
        python -m codes.data.code.generate_ood
        ```
    - Math (alt=1 GSM8K+ variants)
        ```bash
        python -m codes.data.math.generate_ood
        ```
    These scripts map curated IDs to OOD counterparts and save to:
    ```bash
    <data>/<teacher.model>/<dataset>/<student.model>_ood.json
    ```

3. Run interaction experiments
    ```bash
    python -m codes.run_interact
    ```
    This launches a multi-turn student–teacher interaction loop with evaluation.
    Results are saved under:
    ```bash
    experiments/<teacher>/<student>/<dataset>/
    ```

4. Run Self-Guided experiments
    - Collect data
        ```bash
        python -m codes.data.rlhf
        ```
        This collects SFT, DPO data and dumps at 
        ```bash
        <data>/<teacher.model>/<dataset>/<student.model>*dagger*.jsonl
        ```
    - Training
        ```bash
        python -m codes.run_sft_dpo
        ```
        This runs SFT followed by DPO and saves the models at
        ```bash
        experiments/<teacher>/<student>/<dataset>/
        ```
    - Evaluation
        Run interact using the models saved after training (for SGLang/vLLM servers point the `--lora-paths` to the trained checkpoints)

5. Run Peer-Guided experiments
    - Collect data
        ```bash
        python -m codes.data.rlhf_guide
        ```
        This collects SFT, DPO data and dumps at 
        ```bash
        <data>/<teacher.model>/<dataset>/<student.model>_<guide.model>*dagger*.jsonl
        ```
        P.S. we don't need a new script. rlhf and rlhf_guide will be merged soon
    - Training
        ```bash
        python -m codes.run_sft_dpo
        ```
        This runs SFT followed by DPO and saves the models at
        ```bash
        experiments/<teacher>/<student>/<dataset>/
        ```
    - Evaluation
        Run interact using the models saved after training (for SGLang/vLLM servers point the `--lora-paths` to the trained checkpoints)


6. Visualize results
    ```bash
    streamlit run codes/app.py
    ```
    Opens a dashboard to browse:
    - Aggregate pass@k plots
    - Transcripts of teacher ↔ student conversations

---

## 📋 Logging & Timing

- All scripts use Python `logging` (see console for info/debug output).
- Each run logs total time elapsed at the end (hours:minutes:seconds).

---

## 🔎 Notes
- Runs may be compute-intensive; evaluation is often the main bottleneck.
- For reproducibility, seeds for dataset shuffling (e.g., OPC problems) are exposed in config (opc_seed).
- OOD math generation assumes GSM8K+ variants are ordered by id-0, id-1, id-2.