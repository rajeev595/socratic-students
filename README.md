# Socratic Students: Training Language Models to Ask Better Questions for Reasoning

Code for the paper **"Socratic Students: Training Language Models to Ask Better Questions for Reasoning"**
(Rajeev Bhatt Ambati\*, Tianyi Niu\*, Aashu Singh, Shlok Mishra, Snigdha Chaturvedi, Shashank Srivastava; EACL).

A *student* model works on a math or coding problem and may ask a stronger *teacher* model a small
number of questions. The teacher gives guidance but not the answer. After every teacher reply we
measure whether the student can now solve the problem (Pass@k). This repository contains:

- the **interaction framework** and the prompt-based questioning strategies
  (Unscaffolded, Scaffolded, Pre-Assessment, Mid-Assessment);
- **ODQS** (Outcome-Driven Question optimization Strategy): sample candidate questions, score each by
  the student's downstream Pass@k, and train the student on the resulting preferences with DPO
  (Self-ODQS and Peer-ODQS), with no human-labelled question data;
- the **analyses** in the paper (progress, question similarity, teacher leakage, question types,
  generator-vs-solver ablation, human evaluation) and the scripts that draw every figure.

## Contents

- [Installation](#installation)
- [How the code maps to the paper](#how-the-code-maps-to-the-paper)
- [Replicating the paper](#replicating-the-paper)
- [Reproducing each figure and table](#reproducing-each-figure-and-table)
- [Outputs and file formats](#outputs-and-file-formats)
- [Configuration](#configuration)
- [Implementation notes](#implementation-notes)
- [Repository layout](#repository-layout)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Citation](#citation)

## Installation

```bash
git clone <this repository>
cd <this repository>
conda create -n socratic python=3.13 -y && conda activate socratic
pip install -r requirements.txt
```

Models are downloaded from the Hugging Face Hub on first use. Some of them are gated
(Llama-3.3-70B, Mistral-7B-Instruct-v0.3): accept their licences on the Hub and run `huggingface-cli login`.
To keep the model cache somewhere specific, set `HF_HOME` or pass `paths.hf_home=/path/to/cache`.

Figures use LaTeX text rendering, so plotting needs a TeX installation (`latex` on your `PATH`).

**Hardware.** Every model is served locally with [vLLM](https://github.com/vllm-project/vllm); the scripts
start and stop the servers themselves. The paper's runs used 8 x 48 GB GPUs (NVIDIA RTX A6000):
the Qwen2.5-72B teacher on GPUs 0-3 and the 7B student on GPUs 4-7. GPU assignment is set per role
(`teacher.gpus`, `student.gpus`, `peer.gpus`, `judge.gpus`, `evaluator.gpus`) and can be changed on the
command line. All commands are run from the repository root.

## How the code maps to the paper

| Paper | In this repository |
|---|---|
| Student `S`, teacher `T` | `student.model`, `teacher.model` |
| Static baseline | turn 0 of any run (Pass@k before the first question) |
| Unscaffolded | `experiment.scaffolded=False` (default) |
| Scaffolded | `experiment.scaffolded=True` |
| Pre-Assessment | `experiment.assessment_turn=0` |
| Mid-Assessment | `experiment.assessment_turn=2` |
| Self-ODQS | `codes.data.collect_self_odqs` then `codes.run_sft_dpo` with `data.source=self` |
| Peer-ODQS | `codes.data.collect_peer_odqs` then `codes.run_sft_dpo` with `data.source=peer` |
| Rejection Sampling | `codes.data.collect_rejection_sampling` then `codes.run_sft_dpo` |
| Peer model | `peer.model` |
| LLM judge | `judge.model` |
| Question generator vs. solver | `student` (asks the questions) vs. `evaluator` (answers the problem) in `codes.run_cross_eval` |

Models used in the paper, by their nickname in `codes/utils/llm/server.py`:

| Role | Nickname | Model |
|---|---|---|
| Teacher | `qwen72` | Qwen/Qwen2.5-72B-Instruct |
| Students | `qwen7`, `mistral7` | Qwen/Qwen2.5-7B-Instruct, mistralai/Mistral-7B-Instruct-v0.3 |
| Peer | `qwen30` | Qwen/Qwen3-30B-A3B-Instruct-2507 |
| Judges | `qwen72`, `llama70` | Qwen/Qwen2.5-72B-Instruct, meta-llama/Llama-3.3-70B-Instruct |

Any other Hugging Face model id or local path can be used in place of a nickname.

**Turns** are numbered as in the paper. With `experiment.total_turns=5` (N = 5 questioning turns),
every turn `t = 0..5` first evaluates the student (Pass@k) and then lets it ask a question that the
teacher answers. Turn 0 is the static baseline, before any question has been asked. Result files,
collected data and analysis outputs are all keyed by this turn number.

**Experiment names.** `experiment.name` is only a label: it decides where results are written
(`experiments/<name>/<teacher>/<student>/<dataset>/`). The plotting and analysis scripts expect the
names used by the scripts in `scripts/`:

| Name | What it is |
|---|---|
| `unscaffolded`, `scaffolded` | prompt-only strategies |
| `pre_assessment`, `mid_assessment` | Scaffolded + assessment |
| `self_odqs`, `peer_odqs` | ODQS data collection, training checkpoints, and the trained student evaluated without scaffold or assessment |
| `self_odqs_pre_assessment`, `self_odqs_mid_assessment`, `peer_odqs_...` | trained student evaluated with Scaffolded + assessment (the reported ODQS results; Pre for math, Mid for code) |
| `rejection_sampling`, `rejection_sampling_dpo` | LLM-judge baselines |
| `assessment_turn_0` ... `assessment_turn_5` | Unscaffolded + one assessment at turn `t` |
| `self_odqs_assessment_turn_<t>` | trained student, Unscaffolded + assessment at turn `t` |
| `cross_eval/gen_<generator>__eval_<solver>` | generator-vs-solver runs |

## Replicating the paper

Each step is one script. They are parameterised by environment variables:

| Variable | Values | Default |
|---|---|---|
| `STUDENT` | `qwen7`, `mistral7` | `qwen7` |
| `DOMAIN` | `math` (GSM8K), `code` (HumanEval + OpenCoder) | `math` |
| `SOURCE` | `self`, `peer` (step 3 only) | `self` |
| `NUM_GPUS` | GPUs for training | `8` |
| `EXP` | experiments root | `./experiments` |
| `EXTRA` | extra overrides appended to every command, e.g. `EXTRA="teacher.gpus=[0,1] student.gpus=[2,3]"` | empty |

The full paper is four runs of the pipeline: `STUDENT` in {`qwen7`, `mistral7`} x `DOMAIN` in {`math`, `code`}.

```bash
export STUDENT=qwen7 DOMAIN=math

bash scripts/1_curate_data.sh            # rebuild the paper's problem sets from problem_ids/
bash scripts/2_prompting_strategies.sh   # Unscaffolded, Scaffolded, Pre-/Mid-Assessment   (RQ1, RQ2)
SOURCE=self bash scripts/3_odqs.sh       # Self-ODQS: collect preferences, train, evaluate (RQ3)
SOURCE=peer bash scripts/3_odqs.sh       # Peer-ODQS
bash scripts/4_rejection_sampling.sh     # LLM-judge baselines
bash scripts/5_ablations.sh              # assessment position, generator vs. solver
bash scripts/6_analysis.sh               # progress, similarity, leakage, question types
bash scripts/7_figures.sh                # all figures and significance tables (CPU only)
```

Interaction, data-collection and training runs save their progress and resume where they stopped if you run them again.

### Step 1: problem sets

The exact problem sets of the paper are released as ordered lists of problem ids in
[`problem_ids/`](problem_ids/). Rebuild the full problem files from the source datasets with:

```bash
python -m codes.data.build_splits        # writes data/qwen72/<dataset>/splits_0.5/<student>_{train,val,test}.json
```

This needs no GPU and gives the same train / val / test problems, in the same order, as used in the paper.

To curate problem sets yourself (a different student or teacher, or from scratch):

```bash
python -m codes.data.math.sample_hard student.model=qwen7 data.type=math experiment.dataset=gsm8k
python -m codes.data.code.sample_hard student.model=qwen7 data.type=code experiment.dataset=humaneval_opc
```

A problem is kept if the teacher solves it and the student does not (static Pass@5 < 0.5 over samples
at temperature 0.3). Math problems come from GSM8K (`openai/gsm8k`, train split); code problems from
HumanEval (`openai_humaneval`) and OpenCoder (`OpenCoder-LLM/opc-sft-stage2`, `educational_instruct`).
The kept problems are shuffled and split 50 / 20 / 30 into train / val / test, so curating again
gives a different set from the released one. ODQS data is collected on the train and val splits;
everything reported in the paper is measured on the test split.

### Step 2: prompt-based strategies

```bash
C="student.model=qwen7 data.type=math experiment.dataset=gsm8k data.split=test data.data_amount=1000"

python -m codes.run_interact $C experiment.name=unscaffolded
python -m codes.run_interact $C experiment.name=scaffolded     experiment.scaffolded=True
python -m codes.run_interact $C experiment.name=pre_assessment experiment.scaffolded=True experiment.assessment_turn=0
python -m codes.run_interact $C experiment.name=mid_assessment experiment.scaffolded=True experiment.assessment_turn=2
```

### Step 3: ODQS

```bash
C="student.model=qwen7 data.type=math experiment.dataset=gsm8k data.name=odqs data.source=self"

# (a) collect outcome-labelled question preferences
python -m codes.data.collect_self_odqs $C experiment.name=self_odqs data.split=train data.data_amount=1000
python -m codes.data.collect_self_odqs $C experiment.name=self_odqs data.split=val   data.data_amount=500

# (b) train the student with DPO (LoRA)
accelerate launch --multi_gpu --num_processes 8 -m codes.run_sft_dpo $C experiment.name=self_odqs

# (c) evaluate the trained student (Pre-Assessment for math; use assessment_turn=2 for code)
CKPT=$(ls -d experiments/self_odqs/qwen72/qwen7/gsm8k/dpo_0/checkpoint-* | sort -V | head -n1)
python -m codes.run_interact student.model=qwen7 data.type=math experiment.dataset=gsm8k \
    data.split=test data.data_amount=1000 \
    student.lora.name=qwen7_odqs student.lora.path=$CKPT \
    experiment.name=self_odqs_pre_assessment experiment.scaffolded=True experiment.assessment_turn=0
```

For Peer-ODQS use `codes.data.collect_peer_odqs`, `data.source=peer` and `experiment.name=peer_odqs`.

### Steps 4-7

See the scripts; each one lists its commands with a comment per experiment.

## Reproducing each figure and table

After the runs above, `bash scripts/7_figures.sh` writes everything under `experiments/plots/`.

| Paper | Command | Output |
|---|---|---|
| Fig. 2 (RQ1), Fig. 3 (RQ2), Fig. 4 (RQ3) | `python -m codes.plot_results` | `plots/results/*__k5__rq{1,2,3}.pdf` |
| Fig. 10, 11 (Pass@1, Pass@10) | same | `plots/results/*__k{1,10}__rq3.pdf` |
| Tables 3-6 (turn-wise Pass@5) | `python -m codes.plot_results --tables` | printed |
| Tables 7-9 (significance) | `python -m codes.significance` | printed, `plots/significance.json` |
| Table 2 (preference data sizes) | `wc -l data/qwen72/*/odqs/*_dpo_iter0_train.jsonl` | printed |
| Fig. 5 (progress) | `analysis.task=progress`, then `python -m codes.plot_ablations progress` | `plots/ablations/*__progress.pdf` |
| Fig. 6 (question similarity) | `analysis.task=similarity_judge` | `<experiment>/analysis/similarity_judge_*_overall.png` |
| Fig. 7 (assessment position) | `python -m codes.plot_ablations assessment-position` | `plots/ablations/*__assessment_position.pdf` |
| Fig. 8 (generator vs. solver) | `python -m codes.plot_cross_eval` | `plots/cross_eval/` |
| Fig. 9 (question types) | `analysis.task=question_type`, then `python -m codes.plot_qtype` | `<experiment>/analysis/question_type_distribution_*.pdf` |
| Fig. 12 (teacher size) | `RUN_TEACHER_SIZE=1 bash scripts/5_ablations.sh`, then `python -m codes.plot_ablations teacher-size` | `plots/ablations/*__teacher_size.pdf` |
| Table 10 (teacher leakage) | `analysis.task=leakage` | `<experiment>/analysis/leakage_aggregate.json` (`overall.mean`, `overall.std`) |
| Table 11 (human leakage ratings) | `codes.human_eval.extract_leakage_samples`, `codes.human_eval.analyze_leakage_results` | printed, JSON |
| Table 12 (human question preference) | `codes.human_eval.extract_question_preference_samples`, `codes.human_eval.analyze_preference_results` | printed, JSON |

The analyses are run with `codes.run_analysis`, one task at a time:

```bash
python -m codes.run_analysis student.model=qwen7 data.type=math experiment.dataset=gsm8k data.split=test data.data_amount=1000 \
    analysis.task=leakage judge.model=llama70 \
    experiment.name=self_odqs_pre_assessment analysis.compare=self_odqs_pre_assessment analysis.baseline=self_odqs_pre_assessment
```

| `analysis.task` | What it measures | Judge in the paper |
|---|---|---|
| `progress` | how far the student's reasoning has progressed at each turn (0-1) | Qwen2.5-72B |
| `similarity_judge` | similarity between the questions of `analysis.compare` and `analysis.baseline`, for every pair of turns | Qwen2.5-72B |
| `similarity_embedding`, `similarity_rank` | the same with sentence embeddings / a ranking judge (not in the paper) | - |
| `leakage` | how much of the answer each teacher reply reveals (0-1), reported as the mean across turns | Llama-3.3-70B |
| `question_type` | Unfocused / Conceptual / Hypothesis / Grounded label for every student message | Llama-3.3-70B |

`analysis.compare` is the experiment being analysed and `analysis.baseline` the one it is compared
with (set both to the same name for `progress` and `leakage`). Results go to
`experiments/<analysis.compare>/<teacher>/<student>/<dataset>/analysis/`.

**Human evaluation.** `codes/human_eval/` contains the scripts that sample items for the two Amazon
Mechanical Turk studies, the HIT templates shown to workers (`hit_template_*.html`), and the scripts
that compute agreement, correlations and win rates from the downloaded results.

## Outputs and file formats

```
data/<teacher>/<dataset>/
  splits_0.5/<student>_{train,val,test}.json     curated problems: [{id, prompt, solution, ...}]
  odqs/<student>_dpo_iter0_{train,val}.jsonl     Self-ODQS preference pairs
  odqs/<student>__<peer>__dpo_iter0_*.jsonl      Peer-ODQS preference pairs
  odqs/*_sft_*.jsonl, *_grpo_*.jsonl             chosen questions / all scored candidates per turn

experiments/<name>/<teacher>/<student>/<dataset>/
  transcripts_0/<problem id>.json                [{role, index, turn, content}, ...]
  itemized_results_0.json                        {problem id: {results: {turn: {k: Pass@k}}, generations: ...}}
  aggregate_results_0.json                       {turn: {k: [bootstrapped mean, std]}}
  dpo_0/checkpoint-*/                            LoRA checkpoints (training runs only)
  analysis/                                      outputs of codes.run_analysis
  *.log, config.yaml, overrides.yaml             Hydra logs and the exact configuration of the last run
```

A preference pair is one JSON object per line:

```
{"prompt": [chat messages up to this turn], "chosen_response": "...", "rejected_response": "...",
 "chosen_reward": {...}, "rejected_reward": {...}, "problem": {...}, "turn": 0, "iteration": 0}
```

## Configuration

All settings live in [`configs/config.yaml`](configs/config.yaml) (Hydra) and every key is documented
there. Override any of them on the command line with `key=value`. The ones you are most likely to change:

| Key | Meaning |
|---|---|
| `paths.data`, `paths.experiments`, `paths.hf_home` | where data, results and the model cache live |
| `teacher.model`, `student.model`, `peer.model`, `judge.model` | models (nickname, HF id or local path) |
| `*.gpus`, `*.server` | GPUs and server type (`vllm` or `sglang`) for each role |
| `student.lora.name`, `student.lora.path` | serve a trained LoRA adapter on top of the student |
| `data.type`, `experiment.dataset` | `math` + `gsm8k`, or `code` + `humaneval_opc` |
| `data.split`, `data.data_amount` | which problems and how many |
| `gen.k_values`, `gen.pass_k` | Pass@k values to report; the k used for selection and scoring |
| `gen.num_candidates`, `gen.eps`, `gen.dpo_margin_min` | ODQS data collection |
| `train.*` | LoRA and optimisation settings |
| `experiment.num_workers` | problems processed in parallel (lower it if the servers run out of memory) |

## Implementation notes

Details of what the code does, for anyone comparing against the paper's description:

- **Evaluation.** At every turn the student is prompted for the final answer only, given the
  conversation so far, and `min(gen.num_samples, max(gen.k_values))` answers are sampled at
  temperature 0.3. A math answer passes if its last number equals the gold answer; if that check fails,
  the teacher model is asked whether the answer is equivalent to the gold answer (yes/no). Code is
  executed against the problem's unit tests. Reported means and standard deviations are bootstrapped over problems.
- **Assessment.** The student makes two attempts, the teacher gives feedback on each without revealing
  the answer, and the feedback is appended to the conversation before the given turn.
- **ODQS candidates.** `gen.num_candidates` (5) questions are sampled per turn. For Peer-ODQS the pool
  holds 5 questions from the student and 5 from the peer model.
- **ODQS preferences.** The candidate with the highest Pass@5 is preferred over every candidate that
  scores strictly lower. The conversation then continues with the best candidate with probability
  `1 - gen.eps` (0.8) and with a random other candidate otherwise, which adds some diversity to the
  later turns of the collected data.
- **ODQS training.** LoRA (r = 16, alpha = 32, dropout 0.05) on an 8-bit quantised student, DPO with
  beta = 0.2, learning rate 1e-5 with linear decay, batch size 1 per GPU with 8 gradient-accumulation
  steps on 8 GPUs, up to 5 epochs, keeping the checkpoint with the best validation preference
  accuracy. An SFT stage on the chosen questions can be run first with `train.sft_epochs > 0`; the
  default is DPO only.
- **Data is collected Unscaffolded and without assessment**; the trained student is then evaluated
  with the Scaffolded prompt and the assessment that works best for the domain.

## Repository layout

```
configs/config.yaml               all settings
problem_ids/                      ids of the problems in every train / val / test split used in the paper
scripts/                          one script per pipeline stage (see above)
codes/
  run_interact.py                 student-teacher interaction with per-turn evaluation
  run_sft_dpo.py                  ODQS training (optional SFT, then DPO)
  run_cross_eval.py               generator-vs-solver ablation
  run_analysis.py                 LLM-judge analyses
  plot_results.py                 main result figures and tables
  plot_ablations.py               progress, assessment position, teacher size
  plot_cross_eval.py, plot_qtype.py
  significance.py                 paired-bootstrap significance tests
  data/
    build_splits.py               rebuild the paper's problem sets from problem_ids/
    math/sample_hard.py           curate GSM8K problems
    code/sample_hard.py           curate HumanEval + OpenCoder problems
    collect_self_odqs.py          Self-ODQS preference collection
    collect_peer_odqs.py          Peer-ODQS preference collection
    collect_rejection_sampling.py LLM-judge baseline data
  prompts/                        every prompt used (student, teacher, assessment, judges)
  evaluation/                     answer generation and checking, Pass@k, bootstrap
  human_eval/                     MTurk sampling, HIT templates, analysis
  utils/                          model servers, chat wrapper, file helpers
  experimental/                   not used in the paper, see below
```

`codes/experimental/` holds two things we tried and did not use: a GRPO-style trainer over the scored
candidate groups (`train_grpo.py`) and a data collector that scores questions by a short look-ahead
rollout (`collect_self_odqs_rollout.py`). They are kept for anyone who wants to build on them, but
they are not maintained and are not needed for any result in the paper.

## Acknowledgements

Claude Code (Anthropic) was used as a coding assistant for parts of the development of this codebase.

## License

The code is released under the [MIT License](LICENSE). The problems come from GSM8K, HumanEval and
OpenCoder, which are released under the MIT License by their authors.

## Citation

```bibtex
@inproceedings{ambati-niu-socratic-students,
  title     = {Socratic Students: Training Language Models to Ask Better Questions for Reasoning},
  author    = {Ambati, Rajeev Bhatt and Niu, Tianyi and Singh, Aashu and Mishra, Shlok and Chaturvedi, Snigdha and Srivastava, Shashank},
  booktitle = {Proceedings of the Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  note      = {To appear}
}
```
