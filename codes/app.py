from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple, TypedDict, Literal

import pandas as pd
import streamlit as st
import altair as alt

# =========================
# Config & Types
# =========================

st.set_page_config(page_title="Model Interaction Dashboard", layout="wide")

class FileSig(NamedTuple):
    exists: bool
    size: int
    mtime: float

# One experiment layout: teacher -> (student -> [dataset])
LayoutType = Dict[str, Dict[str, List[str]]]

class TranscriptEntry(TypedDict, total=False):
    role: str
    turn: int
    content: str

# Project dir
PROJ_DIR = "/playpen-ssd2/ambati/interact_plus/"

# Experiment and data dir
DEFAULT_EXP_ROOT = str(Path(PROJ_DIR) / "experiments")
DEFAULT_DATA_ROOT = str(Path(PROJ_DIR) / "data")

# =========================
# FS helpers & caching keys
# =========================

def _is_dir(p: Path) -> bool:
    try:
        return p.is_dir()
    except Exception:
        return False

def file_signature(p: Path) -> FileSig:
    try:
        stt = p.stat()
        return FileSig(True, stt.st_size, stt.st_mtime)
    except Exception:
        return FileSig(False, 0, 0.0)

def dir_signature(path: Path) -> Tuple[int, float]:
    if not path.exists():
        return (0, 0.0)
    count = 0
    latest = 0.0
    for p in path.rglob("*"):
        try:
            stt = p.stat()
        except Exception:
            continue
        count += 1
        latest = max(latest, stt.st_mtime)
    return (count, latest)

def _load_json(p: Path) -> Optional[dict]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# =========================
# Name-aware signatures (detect renames/new dirs)
# =========================

def experiments_root_signature(root: Path) -> str:
    """Changes when top-level experiment folder NAMES (or base mtime) change."""
    if not root.exists():
        return "missing-root"
    parts = []
    try:
        parts.append(f"__base__:{int(root.stat().st_mtime)}")
    except Exception:
        pass
    for d in sorted([p for p in root.iterdir() if _is_dir(p)], key=lambda x: x.name):
        try:
            parts.append(f"EXP:{d.name}|{int(d.stat().st_mtime)}")
        except Exception:
            parts.append(f"EXP:{d.name}|0")
    return hashlib.md5("|".join(parts).encode()).hexdigest()

def layout_signature(exp_dir: Path) -> str:
    """Changes when TEACHER/STUDENT/DATASET directory NAMES (or their mtimes) change."""
    if not exp_dir.exists():
        return "missing-exp"
    items = []
    try:
        items.append(f"__exp__:{int(exp_dir.stat().st_mtime)}")
    except Exception:
        pass
    for t in sorted([p for p in exp_dir.iterdir() if _is_dir(p)], key=lambda x: x.name):
        try:
            items.append(f"T:{t.name}|{int(t.stat().st_mtime)}")
        except Exception:
            items.append(f"T:{t.name}|0")
        for s in sorted([p for p in t.iterdir() if _is_dir(p)], key=lambda x: x.name):
            try:
                items.append(f"S:{s.name}|{int(s.stat().st_mtime)}")
            except Exception:
                items.append(f"S:{s.name}|0")
            for d in sorted([p for p in s.iterdir() if _is_dir(p)], key=lambda x: x.name):
                try:
                    items.append(f"D:{d.name}|{int(d.stat().st_mtime)}")
                except Exception:
                    items.append(f"D:{d.name}|0")
    return hashlib.md5("|".join(items).encode()).hexdigest()

# =========================
# Experiment discovery
# =========================

@st.cache_data(show_spinner=False, ttl=10)
def discover_experiments(exp_root: str, root_sig: str) -> List[str]:
    """Return list of experiment folder names under exp_root."""
    base = Path(exp_root)
    if not _is_dir(base):
        return []
    exps = []
    for exp_dir in sorted([p for p in base.iterdir() if _is_dir(p)], key=lambda x: x.name):
        try:
            has_sub = any(_is_dir(tdir) for tdir in exp_dir.iterdir())
        except Exception:
            has_sub = False
        if has_sub:
            exps.append(exp_dir.name)
    return exps

@st.cache_data(show_spinner=False, ttl=10)
def discover_layout(exp_dir: str, sig: str) -> LayoutType:
    """Discover teachers, students, datasets under a SINGLE experiment directory."""
    base = Path(exp_dir)
    layout: LayoutType = {}
    if not _is_dir(base):
        return layout

    for teacher_dir in sorted([p for p in base.iterdir() if _is_dir(p)], key=lambda x: x.name):
        teacher = teacher_dir.name
        students_map: Dict[str, List[str]] = {}
        for student_dir in sorted([p for p in teacher_dir.iterdir() if _is_dir(p)], key=lambda x: x.name):
            student = student_dir.name
            datasets: List[str] = []
            for dataset_dir in sorted([p for p in student_dir.iterdir() if _is_dir(p)], key=lambda x: x.name):
                has_results = (dataset_dir / "aggregate_results.json").exists()
                has_trans   = _is_dir(dataset_dir / "transcripts")
                if has_results or has_trans:
                    datasets.append(dataset_dir.name)
            if datasets:
                students_map[student] = datasets
        if students_map:
            layout[teacher] = students_map
    return layout

@st.cache_data(show_spinner=False, ttl=10)
def load_aggregate_results(exp_dir: str, teacher: str, student: str, dataset: str, sig: FileSig) -> Optional[dict]:
    p = Path(exp_dir) / teacher / student / dataset / "aggregate_results.json"
    return _load_json(p)

@st.cache_data(show_spinner=False, ttl=10)
def list_pids(exp_dir: str, teacher: str, student: str, dataset: str, sig: Tuple[int, float]) -> List[str]:
    tdir = transcripts_dir(exp_dir, teacher, student, dataset)
    if not _is_dir(tdir):
        return []
    pids = [p.stem for p in tdir.glob("*.json")]
    pids = sorted(pids, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s))
    return pids

@st.cache_data(show_spinner=False, ttl=10)
def load_transcript(exp_dir: str, teacher: str, student: str, dataset: str, pid: str, sig: FileSig) -> Optional[List[TranscriptEntry]]:
    p = transcripts_dir(exp_dir, teacher, student, dataset) / f"{pid}.json"
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else None
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=10)
def load_itemized(exp_dir: str, method: str, teacher: str, student: str, dataset: str) -> Optional[dict]:
    p = Path(exp_dir) / method / teacher / student / dataset / "itemized_results.json"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=10)
def load_itemized_experiment(exp_dir: str, teacher: str, student: str, dataset: str) -> Optional[dict]:
    """
    Load itemized results from experiments/<experiment>/<teacher>/<student>/<dataset>/itemized_results.json
    (No method subdirectory.)
    """
    p = Path(exp_dir) / teacher / student / dataset / "itemized_results.json"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def get_pid_list(itemized: dict) -> List[str]:
    return sorted(itemized.keys(), key=lambda x: (not str(x).isdigit(), int(x) if str(x).isdigit() else str(x)))

def pass_at_k(itemized: dict, pid: str, turn: int, k: int) -> Optional[float]:
    try:
        return float(itemized[str(pid)]["results"][str(turn)][str(k)])
    except Exception:
        return None

def discover_methods_for_experiment(exp_root: str, experiment: str, teacher: str, student: str, dataset: str) -> List[str]:
    base = Path(exp_root) / experiment
    if not base.exists():
        return []
    out = []
    for mdir in sorted([p for p in base.iterdir() if p.is_dir()], key=lambda x: x.name):
        f = mdir / student / dataset / "itemized_results.json"
        if f.exists():
            out.append(mdir.name)
    return out

def collect_methods_across_selected(
    exp_root: str,
    selected_experiments: List[str],
    teacher: str,
    student: str,
    dataset: str,
) -> Dict[str, List[str]]:
    """
    Return {method_name: [experiment1, experiment2, ...]} for which
    experiments contain <method>/<teacher>/<student>/<dataset>/itemized_results.json.
    """
    method2exps: Dict[str, List[str]] = {}
    for exp in selected_experiments:
        ms = discover_methods_for_experiment(exp_root, exp, teacher, student, dataset)
        for m in ms:
            method2exps.setdefault(m, []).append(exp)
    return method2exps

def build_delta_table(
    base_exp_dir: str,
    cmp_exp_dir: str,
    teacher: str,
    student: str,
    dataset: str,
    k: int,
    turn: int,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Cross-experiment Δ table (experiment == method).
      baseline from base_exp_dir
      compare  from cmp_exp_dir
    """
    base = load_itemized_experiment(base_exp_dir, teacher, student, dataset)
    meth = load_itemized_experiment(cmp_exp_dir,  teacher, student, dataset)
    if base is None or meth is None:
        return pd.DataFrame(columns=["pid","baseline_pass","method_pass","delta","bucket"])

    pids = sorted(set(get_pid_list(base)) & set(get_pid_list(meth)),
                  key=lambda x: int(x) if str(x).isdigit() else str(x))
    rows = []
    for pid in pids:
        b = pass_at_k(base, pid, turn, k)
        m = pass_at_k(meth, pid, turn, k)
        if b is None or m is None:
            continue
        d = m - b
        if d >= threshold:
            bucket = "Improved"
        elif d <= -threshold:
            bucket = "Regressed"
        else:
            bucket = "Unchanged"
        rows.append({
            "pid": str(pid),
            "baseline_pass": b,
            "method_pass": m,
            "delta": d,
            "bucket": bucket,
        })
    if not rows:
        return pd.DataFrame(columns=["pid","baseline_pass","method_pass","delta","bucket"])
    return pd.DataFrame(rows).sort_values(["bucket","delta","pid"], ascending=[True, False, True]).reset_index(drop=True)

def extract_student_at_turn(entries: List[TranscriptEntry], turn: int, teacher: str, student: str) -> str:
    for e in entries:
        if e.get("turn") == turn:
            who, _, _ = model_role_for_chat(e.get("role",""), turn, teacher, student)
            if who == "student":
                return e.get("content","") or ""
    return ""

# =========================
# Data root helpers (for showing the question in Transcripts)
# =========================

def data_index_path(data_root: str, teacher: str, dataset: str, student: str) -> Path:
    # data/<teacher>/<dataset>/<student>.json
    return Path(data_root) / teacher / dataset / f"{student}.json"

@st.cache_data(show_spinner=False, ttl=10)
def load_problem_index(data_file: str, sig: FileSig) -> Dict[int, str]:
    """
    Load {id -> prompt} from the dataset index JSON:
    {
      "collected": [ {"id": 1117, "prompt": "...", "solution": ... }, ... ]
    }
    """
    p = Path(data_file)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        items = obj.get("collected", [])
        mapping: Dict[int, str] = {}
        for it in items:
            try:
                mapping[int(it["problem"]["id"])] = str(it["problem"]["prompt"])
            except Exception:
                continue
        return mapping
    except Exception:
        return {}

def lookup_question(data_root: str, teacher: str, dataset: str, student: str, pid_str: str) -> Optional[str]:
    p = data_index_path(data_root, teacher, dataset, student)
    sig = file_signature(p)
    if not sig.exists:
        return None
    index = load_problem_index(str(p), sig)
    try:
        pid_int = int(pid_str)
    except Exception:
        return None
    return index.get(pid_int)

# Detect if a question looks like a Python coding prompt/snippet (signature, imports, doctest, etc.)
def looks_like_python_snippet(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if "```" in t:
        return True
    if t.startswith(("def ", "class ", "from ", "import ")):
        return True
    markers = ["\ndef ", "\nclass ", " from ", " import ", ">>> ", "@dataclass", "if __name__ == '__main__':"]
    score = sum(m in t for m in markers)
    codey_lines = sum(1 for ln in t.splitlines() if ln.strip().startswith(("def ", "class ", "from ", "import ", "#")) or ln.strip().endswith(":"))
    return score >= 2 or codey_lines >= 2

# =========================
# Data shaping & plotting
# =========================

def agg_to_long_df(agg: dict, student: str, experiment: str) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    for turn_str, ks in agg.items():
        try:
            turn = int(turn_str)
        except Exception:
            continue
        for k_str, vals in ks.items():
            try:
                k = int(k_str)
            except Exception:
                continue
            if not isinstance(vals, list) or len(vals) != 2:
                continue
            mean, std = vals
            rows.append({
                "turn": turn,
                "k": k,
                "mean": float(mean),
                "std": float(std),
                "student": student,
                "experiment": experiment,
            })
    if not rows:
        return pd.DataFrame(columns=["turn", "k", "mean", "std", "student", "experiment"])
    df = pd.DataFrame(rows)
    return df.sort_values(["k", "turn", "student", "experiment"]).reset_index(drop=True)

def build_results_df_across_experiments(exp_root: str, experiments: List[str], teacher: str, dataset: str) -> pd.DataFrame:
    """Combine all selected experiments; keep student + experiment columns."""
    frames: List[pd.DataFrame] = []
    for exp in experiments:
        exp_dir = str(Path(exp_root) / exp)
        lay = discover_layout(exp_dir, layout_signature(Path(exp_dir)))
        if teacher not in lay:
            continue
        # students that have this dataset in this experiment
        exp_students = sorted([s for s, ds in lay[teacher].items() if dataset in ds])
        for s in exp_students:
            p = Path(exp_dir) / teacher / s / dataset / "aggregate_results.json"
            sig = file_signature(p)
            agg = load_aggregate_results(exp_dir, teacher, s, dataset, sig)
            if agg:
                frames.append(agg_to_long_df(agg, s, exp))
    if not frames:
        return pd.DataFrame(columns=["turn", "k", "mean", "std", "student", "experiment"])
    return pd.concat(frames, ignore_index=True)

def okabe_ito_palette(n: int) -> List[str]:
    """Color-blind friendly (Okabe-Ito)."""
    base = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    if n <= len(base):
        return base[:n]
    # Extend conservatively if needed
    extra = ["#999999", "#FFD700", "#8B4513", "#708090"]
    return (base + extra)[:n]

def experiment_shape_range(n: int) -> List[str]:
    """
    Distinct point marker shapes per experiment (color-blind conscious).
    Altair supports:
      'circle','square','cross','diamond','triangle-up','triangle-down','triangle-right','triangle-left'
    """
    shapes = ["circle", "square", "triangle-up", "diamond", "cross", "triangle-down", "triangle-right", "triangle-left"]
    if n <= len(shapes):
        return shapes[:n]
    # Repeat with slight reordering if many experiments
    rep = (shapes * ((n + len(shapes) - 1) // len(shapes)))[:n]
    return rep

def plot_passk_per_model(df_all: pd.DataFrame, student: str, k_val: int, exp_colors: Dict[str, str], exp_shapes: Dict[str, str], title_suffix: str = "") -> alt.Chart:
    sub = df_all[(df_all["student"] == student) & (df_all["k"] == k_val)].copy()
    if sub.empty:
        return alt.Chart(pd.DataFrame({"turn": [], "mean": [], "experiment": []})).mark_line().properties(
            title=f"{student} — pass@{k_val} {title_suffix}".strip(), height=300
        )

    # Per-experiment color & shape
    exp_list = sorted(sub["experiment"].unique().tolist())
    color_scale = alt.Scale(domain=exp_list, range=[exp_colors[e] for e in exp_list])
    shape_scale = alt.Scale(domain=exp_list, range=[exp_shapes[e] for e in exp_list])

    base = (
        alt.Chart(sub)
        .transform_calculate(
            mean_c="min(1, max(0, datum.mean))",
        )
        .encode(
            x=alt.X("turn:Q", title="Turn", axis=alt.Axis(values=[1, 3, 5, 7, 9, 11])),
            y=alt.Y("mean_c:Q", title=f"pass@{k_val}", scale=alt.Scale(domain=[0, 1], nice=False)),
            color=alt.Color("experiment:N", title="Experiment", scale=color_scale),
        )
    )

    lines = base.mark_line()
    points = base.mark_point(filled=True, size=70).encode(
        shape=alt.Shape("experiment:N", title="Experiment", scale=shape_scale),
        tooltip=[
            alt.Tooltip("experiment:N", title="Experiment"),
            alt.Tooltip("turn:Q", title="Turn"),
            alt.Tooltip("mean_c:Q", title="mean", format=".3f"),
        ],
    )

    chart = alt.layer(lines, points).properties(
        title=f"{student} — pass@{k_val} {title_suffix}".strip(), height=300
    ).interactive()

    return chart

def transcripts_dir(exp_dir: str, teacher: str, student: str, dataset: str) -> Path:
    return Path(exp_dir) / teacher / student / dataset / "transcripts"

def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    for tok in ["openai/", "meta/", "google/", "qwen/"]:
        s = s.replace(tok, "")
    for ch in ["/", "_", "-", " "]:
        s = s.replace(ch, "")
    return s

def model_role_for_chat(entry_role: str, turn: Optional[int], teacher: str, student: str) -> Tuple[Literal["teacher", "student"], str, str]:
    er = _norm(entry_role); t = _norm(teacher); s = _norm(student)
    if er and er == t: return "teacher", entry_role or teacher, "🧑‍🏫"
    if er and er == s: return "student", entry_role or student, "🧑‍🎓"
    if isinstance(turn, int):
        return ("teacher", entry_role or teacher, "🧑‍🏫") if turn % 2 == 0 else ("student", entry_role or student, "🧑‍🎓")
    is_teacher = bool(er and t and (er in t or t in er))
    is_student = bool(er and s and (er in s or s in er))
    if is_teacher and not is_student: return "teacher", entry_role or teacher, "🧑‍🏫"
    if is_student and not is_teacher: return "student", entry_role or student, "🧑‍🎓"
    return "teacher", entry_role or teacher, "🧑‍🏫"

# =========================
# Sidebar (Teacher/Dataset + EXPERIMENT selection)
# =========================

st.sidebar.title("Controls")
exp_root = DEFAULT_EXP_ROOT
data_root = DEFAULT_DATA_ROOT

root_sig = experiments_root_signature(Path(exp_root))
all_experiments = discover_experiments(exp_root, root_sig)
if not all_experiments:
    st.sidebar.warning("No experiments found under the given root.")
    st.title("Model Interaction Dashboard")
    st.stop()

# Build a union layout across experiments to populate Teacher/Dataset/Students
def union_layout_for_controls(exp_root: str, experiments: List[str]) -> LayoutType:
    union: LayoutType = {}
    for exp in experiments:
        exp_dir = str(Path(exp_root) / exp)
        lay = discover_layout(exp_dir, layout_signature(Path(exp_dir)))
        for teacher, stu_map in lay.items():
            union.setdefault(teacher, {})
            for student, datasets in stu_map.items():
                union[teacher].setdefault(student, [])
                for d in datasets:
                    if d not in union[teacher][student]:
                        union[teacher][student].append(d)
    return union

union_layout = union_layout_for_controls(exp_root, all_experiments)
if not union_layout:
    st.sidebar.warning("No teachers/students/datasets found across experiments.")
    st.title("Model Interaction Dashboard")
    st.stop()

all_teachers = sorted(union_layout.keys())
teacher = st.sidebar.selectbox("Teacher", all_teachers, key="sidebar_teacher")

datasets_for_teacher = sorted({d for s, ds in union_layout[teacher].items() for d in ds})
if not datasets_for_teacher:
    st.sidebar.error("No datasets found for the selected teacher.")
    st.stop()
preferred = "math"
default_idx = datasets_for_teacher.index(preferred) if preferred in datasets_for_teacher else 0
dataset = st.sidebar.selectbox("Dataset", datasets_for_teacher, index=default_idx, key="sidebar_dataset")

# NEW: Experiment multiselect (replaces student controls)
selected_experiments = st.sidebar.multiselect(
    "Experiments (included in plots)",
    options=all_experiments,
    default=all_experiments,
    key="sidebar_experiments_multiselect",
)

if not selected_experiments:
    st.sidebar.info("Select at least one experiment to visualize results.")
    st.stop()

# Build the combined results across selected experiments
df_all = build_results_df_across_experiments(exp_root, selected_experiments, teacher, dataset)

# Build the combined results across selected experiments
df_all = build_results_df_across_experiments(exp_root, selected_experiments, teacher, dataset)

# Prefer students from df_all when present; otherwise fall back to filesystem discovery
if not df_all.empty:
    students_union = sorted(df_all["student"].unique().tolist())
else:
    # Union across selected experiments via discover_layout (doesn't require aggregate_results.json)
    students_set = set()
    for exp in selected_experiments:
        exp_dir = str(Path(exp_root) / exp)
        lay = discover_layout(exp_dir, layout_signature(Path(exp_dir)))
        if teacher in lay:
            for s, ds in lay[teacher].items():
                if dataset in ds:
                    students_set.add(s)
    students_union = sorted(students_set)

k_options = sorted(df_all["k"].unique().tolist()) if not df_all.empty else [1, 5, 10]
default_k = 5 if 5 in k_options else (k_options[0] if k_options else 1)
selected_k = st.radio(
    "Pass@k metric",
    options=k_options,
    index=k_options.index(default_k) if k_options else 0,
    horizontal=True,
    key="results_k_radio",
)

# =========================
# Main Tabs
# =========================

tab_results, tab_transcripts, tab_analysis = st.tabs(["📈 Results (Per Model)", "💬 Transcripts", "🧪 Analysis"])

# =========================
# Results Tab — PER MODEL rows, each showing all experiments
# =========================

with tab_results:
    st.subheader("Aggregate Results (per model)")
    st.caption(f"Teacher: `{teacher}` · Dataset: `{dataset}` · Experiments: {', '.join(selected_experiments) if selected_experiments else '(none)' }")

    if df_all.empty or not students_union:
        st.info("No matching results for the current selection.")
    else:
        # Prepare consistent color/shape for experiments
        exps_sorted = sorted(set(selected_experiments) & set(df_all["experiment"].unique().tolist()))
        colors = okabe_ito_palette(len(exps_sorted))
        shapes = experiment_shape_range(len(exps_sorted))
        exp_colors = {e: colors[i] for i, e in enumerate(exps_sorted)}
        exp_shapes = {e: shapes[i] for i, e in enumerate(exps_sorted)}

        title_suffix = f"({teacher}, {dataset})"
        for stu in students_union:
            st.markdown(f"### `{stu}`")
            with st.expander("Show data table", expanded=False):
                st.dataframe(
                    df_all[(df_all["student"] == stu) & (df_all["k"] == int(selected_k))].sort_values(["turn", "experiment"]),
                    width='stretch'
                )

            # Show only the chosen Pass@k
            st.altair_chart(
                plot_passk_per_model(df_all, stu, int(selected_k), exp_colors, exp_shapes, title_suffix),
                use_container_width=True
            )
            st.markdown("---")

# =========================
# Transcripts Tab — unchanged (experiment select inside tab)
# =========================

with tab_transcripts:
    st.subheader("Conversation Viewer")
    st.caption("Pick an experiment here (not in the sidebar).")

    transcript_experiment = st.selectbox("Experiment", all_experiments, index=0, key="transcript_experiment")

    exp_dir_for_transcripts = str(Path(exp_root) / transcript_experiment)
    layout_for_exp = discover_layout(exp_dir_for_transcripts, layout_signature(Path(exp_dir_for_transcripts)))

    if teacher not in layout_for_exp:
        st.info(f"No data for teacher `{teacher}` in experiment `{transcript_experiment}`.")
    else:
        t_students = sorted([s for s, ds in layout_for_exp[teacher].items() if dataset in ds])
        if not t_students:
            st.info(f"No students with dataset `{dataset}` for teacher `{teacher}` in `{transcript_experiment}`.")
        else:
            t_student = st.selectbox("Student", t_students, index=0, key="transcript_student")

            tdir = transcripts_dir(exp_dir_for_transcripts, teacher, t_student, dataset)
            t_sig = dir_signature(tdir)
            pids = list_pids(exp_dir_for_transcripts, teacher, t_student, dataset, t_sig)

            if not pids:
                st.info("No transcripts found (no JSON files in transcripts/).")
            else:
                pid = st.selectbox("Problem ID (pid)", pids, index=0, key="transcript_pid")

                # --- Show the question at the top (from data/<teacher>/<dataset>/<student>.json) ---
                question = lookup_question(DEFAULT_DATA_ROOT, teacher, dataset, t_student, pid)
                st.markdown("#### Question")
                if question:
                    if looks_like_python_snippet(question):
                        st.code(question.strip(), language="python")
                    else:
                        st.markdown(f"> {question.strip('#').replace("$", "\\$")}")
                else:
                    st.markdown("> _Not found in data index._")

                this_file = tdir / f"{pid}.json"
                f_sig = file_signature(this_file)
                data = load_transcript(exp_dir_for_transcripts, teacher, t_student, dataset, pid, f_sig)

                if not isinstance(data, list):
                    st.warning("Transcript JSON could not be loaded or has unexpected format.")
                else:
                    st.write(
                        f"**Experiment:** `{transcript_experiment}` · **Teacher:** `{teacher}` · "
                        f"**Student:** `{t_student}` · **Dataset:** `{dataset}` · **pid:** `{pid}`"
                    )

                    # Minimal CSS for chat bubbles
                    st.markdown(
                        """
                        <style>
                        .chat-row { display: flex; width: 100%; margin: 0.10rem 0; }
                        .chat-left  { justify-content: flex-start; }
                        .chat-right { justify-content: flex-end;   }
                        .bubble-wrap { display: flex; align-items: flex-start; gap: 0.4rem; max-width: 70%; }
                        .wrap-left  { flex-direction: row; }
                        .wrap-right { flex-direction: row-reverse; }
                        .avatar-emoji { font-size: 3.6rem; line-height: 1; margin-top: 2px; }
                        .bubble { padding: 0.4rem 0.7rem; border-radius: 12px; line-height: 1.35;
                                  border: 1px solid rgba(0,0,0,0.06); box-shadow: 0 1px 2px rgba(0,0,0,0.04);
                                  word-wrap: break-word; white-space: pre-wrap; font-size: 1rem; }
                        .bubble-teacher { background: #f1f5ff; }
                        .bubble-student { background: #f7f7f7; }
                        .meta { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.15rem; }
                        .name { font-weight: 700; }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Render chat
                    for entry in data:
                        who, display_name, avatar = model_role_for_chat(
                            entry.get("role", ""),
                            entry.get("turn", None),
                            teacher,
                            t_student
                        )
                        content = entry.get("content", "") or ""
                        is_teacher = (who == "teacher")
                        row_class  = "chat-left" if is_teacher else "chat-right"
                        wrap_class = "wrap-left" if is_teacher else "wrap-right"
                        bubble_cls = "bubble-teacher" if is_teacher else "bubble-student"

                        st.markdown(
                            f"""
                            <div class="chat-row {row_class}">
                              <div class="bubble-wrap {wrap_class}">
                                <div class="avatar"><span class="avatar-emoji">{avatar}</span></div>
                                <div class="bubble {bubble_cls}">
                                  <div class="meta"><span class="name">{display_name}</span></div>
                                  <div>{content}</div>
                                </div>
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

# =========================
# Analysis Tab — unchanged (experiment select inside tab)
# =========================

with tab_analysis:
    st.subheader("Qualitative Analysis")
    st.caption("Pick a Baseline experiment and a Compare experiment. We compute Δ = compare − baseline.")

    if not students_union:
        st.info("No students found for the current selection (teacher/dataset) across the chosen experiments.")
        st.stop()

    # --- Who/what ---
    top_cols = st.columns(2)
    with top_cols[0]:
        a_student = st.selectbox("Student", students_union, key="analysis_student")
    with top_cols[1]:
        k_opts = sorted(df_all[df_all["student"] == a_student]["k"].unique().tolist()) if not df_all.empty else [1,5,10]
        a_k = st.selectbox("Pass@k", k_opts, index=0, key="analysis_k")

    mid_cols = st.columns(3)
    with mid_cols[0]:
        t_opts = sorted(df_all[df_all["student"] == a_student]["turn"].unique().tolist()) if not df_all.empty else [2,3,4,5,6,7,8,9,10,11]
        a_turn = st.selectbox("Turn", t_opts, index=max(0, len(t_opts)-1), key="analysis_turn")

    # --- Threshold: slider + plain text box (no +/-), synced ---
    if "analysis_thr_slider" not in st.session_state:
        st.session_state.analysis_thr_slider = 1.0
    if "analysis_thr_box_str" not in st.session_state:
        st.session_state.analysis_thr_box_str = f"{st.session_state.analysis_thr_slider:.2f}"

    def _sync_from_slider():
        st.session_state.analysis_thr_box_str = f"{st.session_state.analysis_thr_slider:.2f}"

    def _sync_from_box():
        s = (st.session_state.analysis_thr_box_str or "").strip()
        try:
            v = float(s)
        except Exception:
            return
        v = max(0.0, min(1.0, v))
        st.session_state.analysis_thr_slider = v
        st.session_state.analysis_thr_box_str = f"{v:.2f}"

    with mid_cols[1]:
        st.slider(
            "Δ threshold",
            min_value=0.0, max_value=1.0, step=0.05,
            value=st.session_state.analysis_thr_slider,
            key="analysis_thr_slider",
            on_change=_sync_from_slider,
            help="Minimum Δ = (compare − baseline) to mark as Improved (≤ −Δ as Regressed)."
        )
    with mid_cols[2]:
        st.text_input(
            label="",
            value=st.session_state.analysis_thr_box_str,
            key="analysis_thr_box_str",
            on_change=_sync_from_box,
            placeholder="e.g., 0.50",
            label_visibility="collapsed",
        )
    a_thr = float(st.session_state.analysis_thr_slider)

    # --- Baseline & Compare (EXPERIMENT == METHOD) ---
    st.markdown("#### Baseline & Compare (experiments)")

    if not selected_experiments:
        st.info("Select at least one experiment in the sidebar.")
        st.stop()

    # Pick two experiments from the sidebar selection
    bc_cols = st.columns(2)
    with bc_cols[0]:
        baseline_exp = st.selectbox("Baseline", selected_experiments, index=0, key="analysis_baseline_exp")
    with bc_cols[1]:
        cmp_default_idx = 1 if len(selected_experiments) > 1 else 0
        compare_exp = st.selectbox("Compare", selected_experiments, index=cmp_default_idx, key="analysis_compare_exp")

    # Compute Δ across the two experiments
    exp_dir_base = str(Path(exp_root) / baseline_exp)
    exp_dir_cmp  = str(Path(exp_root) / compare_exp)

    adf = build_delta_table(
        exp_dir_base, exp_dir_cmp, teacher, a_student, dataset,
        int(a_k), int(a_turn), float(a_thr)
    )

    if adf.empty:
        st.info("No overlapping examples between the two experiments (or missing itemized files).")
        st.stop()

    total = len(adf)
    n_imp = int((adf["bucket"]=="Improved").sum())
    n_reg = int((adf["bucket"]=="Regressed").sum())
    n_unch = total - n_imp - n_reg

    st.caption(f"Baseline uses `{baseline_exp}`  |  Compare uses `{compare_exp}`")
    st.markdown(
        f"**Total:** {total} · **Improved:** {n_imp} ({(100*n_imp/total):.1f}%) · "
        f"**Regressed:** {n_reg} ({(100*n_reg/total):.1f}%) · **Unchanged:** {n_unch} ({(100*n_unch/total):.1f}%)"
    )

    # Bucket filter + preview text
    bucket_pick = st.radio("Bucket", ["Improved","Regressed","Unchanged","All"], horizontal=True, key="analysis_bucket")
    show_df = adf if bucket_pick == "All" else adf[adf["bucket"]==bucket_pick].copy()

    def _preview_question(pid: str) -> str:
        q = lookup_question(DEFAULT_DATA_ROOT, teacher, dataset, a_student, pid)
        if not q:
            return ""
        q = q.strip()
        return (q[:140] + "…") if len(q) > 160 else q

    show_df = show_df.copy()
    show_df["question_preview"] = show_df["pid"].apply(_preview_question)
    show_df = show_df[["pid","bucket","delta","baseline_pass","method_pass","question_preview"]]
    st.dataframe(show_df, use_container_width=True)

    st.divider()
    st.markdown("### Inspect example")

    pid_choices = show_df["pid"].tolist()
    a_pid = st.selectbox("Problem ID", pid_choices, index=0, key="analysis_pid")

    # --- Question ---
    st.markdown("#### Question")
    qtext = lookup_question(DEFAULT_DATA_ROOT, teacher, dataset, a_student, a_pid)
    if qtext:
        if looks_like_python_snippet(qtext):
            st.code(qtext.strip(), language="python")
        else:
            st.markdown(f"> {qtext.strip('#').replace('$', '\\$')}")
    else:
        st.markdown("> _Not found in data index._")

    # --- Full transcripts from BOTH experiments ---
    base_tdir = transcripts_dir(exp_dir_base, teacher, a_student, dataset)
    cmp_tdir  = transcripts_dir(exp_dir_cmp,  teacher, a_student, dataset)

    base_fp = base_tdir / f"{a_pid}.json"
    cmp_fp  = cmp_tdir  / f"{a_pid}.json"

    base_tdata = load_transcript(exp_dir_base, teacher, a_student, dataset, a_pid, file_signature(base_fp))
    cmp_tdata  = load_transcript(exp_dir_cmp,  teacher, a_student, dataset, a_pid, file_signature(cmp_fp))

    base_pass = float(adf.loc[adf["pid"]==a_pid, "baseline_pass"].iloc[0])
    meth_pass = float(adf.loc[adf["pid"]==a_pid, "method_pass"].iloc[0])
    delta_val = float(adf.loc[adf["pid"]==a_pid, "delta"].iloc[0])
    st.markdown(
        f"**Baseline pass@{int(a_k)}:** {base_pass:.0f} &nbsp;&nbsp; "
        f"**Compare pass@{int(a_k)}:** {meth_pass:.0f} &nbsp;&nbsp; "
        f"**Δ (compare − baseline):** {delta_val:+.0f} &nbsp;&nbsp; "
        f"**Turn:** {int(a_turn)}"
    )

    # Bubble CSS (reuse)
    st.markdown(
        """
        <style>
        .chat-row { display: flex; width: 100%; margin: 0.10rem 0; }
        .chat-left  { justify-content: flex-start; }
        .chat-right { justify-content: flex-end;   }
        .bubble-wrap { display: flex; align-items: flex-start; gap: 0.4rem; max-width: 70%; }
        .wrap-left  { flex-direction: row; }
        .wrap-right { flex-direction: row-reverse; }
        .avatar-emoji { font-size: 3.6rem; line-height: 1; margin-top: 2px; }
        .bubble { padding: 0.4rem 0.7rem; border-radius: 12px; line-height: 1.35;
                  border: 1px solid rgba(0,0,0,0.06); box-shadow: 0 1px 2px rgba(0,0,0,0.04);
                  word-wrap: break-word; white-space: pre-wrap; font-size: 1rem; }
        .bubble-teacher { background: #f1f5ff; }
        .bubble-student { background: #f7f7f7; }
        .meta { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.15rem; }
        .name { font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Full Transcripts")
    with st.expander(f"Baseline transcript — {baseline_exp}", expanded=False):
        if not isinstance(base_tdata, list):
            st.warning("Baseline transcript not found or invalid format.")
        else:
            for entry in base_tdata:
                who, display_name, avatar = model_role_for_chat(entry.get("role",""), entry.get("turn", None), teacher, a_student)
                content = entry.get("content","") or ""
                is_teacher = (who == "teacher")
                row_class  = "chat-left" if is_teacher else "chat-right"
                wrap_class = "wrap-left" if is_teacher else "wrap-right"
                bubble_cls = "bubble-teacher" if is_teacher else "bubble-student"
                st.markdown(
                    f"""
                    <div class="chat-row {row_class}">
                      <div class="bubble-wrap {wrap_class}">
                        <div class="avatar"><span class="avatar-emoji">{avatar}</span></div>
                        <div class="bubble {bubble_cls}">
                          <div class="meta"><span class="name">{display_name}</span></div>
                          <div>{content}</div>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with st.expander(f"Compare transcript — {compare_exp}", expanded=False):
        if not isinstance(cmp_tdata, list):
            st.warning("Compare transcript not found or invalid format.")
        else:
            for entry in cmp_tdata:
                who, display_name, avatar = model_role_for_chat(entry.get("role",""), entry.get("turn", None), teacher, a_student)
                content = entry.get("content","") or ""
                is_teacher = (who == "teacher")
                row_class  = "chat-left" if is_teacher else "chat-right"
                wrap_class = "wrap-left" if is_teacher else "wrap-right"
                bubble_cls = "bubble-teacher" if is_teacher else "bubble-student"
                st.markdown(
                    f"""
                    <div class="chat-row {row_class}">
                      <div class="bubble-wrap {wrap_class}">
                        <div class="avatar"><span class="avatar-emoji">{avatar}</span></div>
                        <div class="bubble {bubble_cls}">
                          <div class="meta"><span class="name">{display_name}</span></div>
                          <div>{content}</div>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
