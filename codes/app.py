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
PROJ_DIR = "/playpen-ssd/ambati/interact_plus/"

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
                mapping[int(it["id"])] = str(it["prompt"])
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
    # Quick markers for code-like prompts
    if t.startswith(("def ", "class ", "from ", "import ")):
        return True
    markers = ["\ndef ", "\nclass ", " from ", " import ", ">>> ", "@dataclass", "if __name__ == '__main__':"]
    score = sum(m in t for m in markers)
    # Also count "codey" lines
    codey_lines = sum(1 for ln in t.splitlines() if ln.strip().startswith(("def ", "class ", "from ", "import ", "#")) or ln.strip().endswith(":"))
    return score >= 2 or codey_lines >= 2

# =========================
# Data shaping & plotting
# =========================

def agg_to_long_df(agg: dict, student: str) -> pd.DataFrame:
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
            rows.append({"turn": turn, "k": k, "mean": float(mean), "std": float(std), "student": student})
    if not rows:
        return pd.DataFrame(columns=["turn", "k", "mean", "std", "student"])
    df = pd.DataFrame(rows)
    return df.sort_values(["k", "turn", "student"]).reset_index(drop=True)

def build_results_df(exp_dir: str, teacher: str, dataset: str, students: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for s in students:
        p = Path(exp_dir) / teacher / s / dataset / "aggregate_results.json"
        sig = file_signature(p)
        agg = load_aggregate_results(exp_dir, teacher, s, dataset, sig)
        if agg:
            frames.append(agg_to_long_df(agg, s))
    if not frames:
        return pd.DataFrame(columns=["turn", "k", "mean", "std", "student"])
    return pd.concat(frames, ignore_index=True)

def make_palette(n: int) -> List[str]:
    """
    Produce n distinct colors; start with D3 category10, then extend using HSV if needed.
    """
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    if n <= len(base):
        return base[:n]
    cols = base[:]
    import colorsys
    extra = n - len(base)
    for i in range(extra):
        h = (i / max(1, extra))
        s, v = 0.65, 0.65
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append("#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255)))
    return cols

def build_student_colors(all_students_for_teacher_dataset: List[str]) -> Dict[str, str]:
    ordered = sorted(all_students_for_teacher_dataset)
    palette = make_palette(len(ordered))
    return {stu: palette[i] for i, stu in enumerate(ordered)}

def plot_passk(df: pd.DataFrame, k_val: int, student_colors: Dict[str, str], title_suffix: str = "") -> alt.Chart:
    sub = df[df["k"] == k_val].copy()
    if sub.empty:
        return alt.Chart(pd.DataFrame({"turn": [], "mean": [], "student": []})).mark_line().properties(
            title=f"pass@{k_val} {title_suffix}".strip(), height=300
        )

    # Color domain only for students present, with safe fallbacks
    domain = sorted(sub["student"].unique().tolist())
    fallback = make_palette(len(domain))
    mapped = [student_colors.get(s, fallback[i]) for i, s in enumerate(domain)]
    color_scale = alt.Scale(domain=domain, range=mapped)

    # Clamp into [0, 1] and lock axis domain
    base = (
        alt.Chart(sub)
        .transform_calculate(
            mean_c="min(1, max(0, datum.mean))",
            upper="min(1, max(0, datum.mean + datum.std))",
            lower="min(1, max(0, datum.mean - datum.std))",
        )
    )

    band = (
        base.mark_area(opacity=0.2)
        .encode(
            x=alt.X("turn:Q", title="Turn"),
            y=alt.Y("lower:Q", title=f"pass@{k_val}", scale=alt.Scale(domain=[0, 1], nice=False)),
            y2="upper:Q",
            color=alt.Color("student:N", legend=None, scale=color_scale),
        )
    )

    lines = (
        base.mark_line(point=True)
        .encode(
            x=alt.X("turn:Q", title="Turn"),
            y=alt.Y("mean_c:Q", title=f"pass@{k_val}", scale=alt.Scale(domain=[0, 1], nice=False)),
            color=alt.Color("student:N", title="Student", scale=color_scale),
            tooltip=[
                "student",
                "turn",
                alt.Tooltip("mean_c:Q", title="mean", format=".3f"),
                alt.Tooltip("std:Q", format=".3f"),
                alt.Tooltip("lower:Q", format=".3f"),
                alt.Tooltip("upper:Q", format=".3f"),
            ],
        )
    )

    return (band + lines).properties(title=f"pass@{k_val} {title_suffix}".strip(), height=300).interactive()

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

def emoji_for_index(i: int) -> str:
    squares = ["🟦","🟧","🟩","🟥","🟪","🟨","⬛","⬜"]
    return squares[i % len(squares)]

def build_student_emoji_labels(students: List[str]) -> Dict[str, str]:
    ordered = sorted(students)
    return {stu: f"{emoji_for_index(i)} {stu}" for i, stu in enumerate(ordered)}

# =========================
# Sidebar (Teacher/Dataset + combined multiselect)
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

# Build a union layout across experiments just to populate Teacher/Dataset/Students
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

# Prefer "math" as initial default; fall back to alphabetical if absent
datasets_for_teacher = sorted({d for s, ds in union_layout[teacher].items() for d in ds})
if not datasets_for_teacher:
    st.sidebar.error("No datasets found for the selected teacher.")
    st.stop()
preferred = "math"
default_idx = datasets_for_teacher.index(preferred) if preferred in datasets_for_teacher else 0
dataset = st.sidebar.selectbox("Dataset", datasets_for_teacher, index=default_idx, key="sidebar_dataset")

# Union of students for the chosen (teacher, dataset) across all experiments
students_union = sorted([s for s, ds in union_layout[teacher].items() if dataset in ds])

# Auto-merge newcomers into the selection so they appear without clicks
prev_sel = st.session_state.get("selected_students", [])
if prev_sel:
    merged = [s for s in prev_sel if s in students_union] + [s for s in students_union if s not in prev_sel]
else:
    merged = students_union[:]
st.session_state.selected_students = merged

# Color palette for plots
student_colors = build_student_colors(students_union)

# Emoji markers & option labels aligned with sorted(students_union)
emoji_map = {stu: emoji_for_index(i) for i, stu in enumerate(sorted(students_union))}
options_map = {f"{emoji_map[s]} {s}": s for s in students_union}

# Keep widget selection in sync with auto-merged students
labels_for_selected = [f"{emoji_map[s]} {s}" for s in st.session_state.selected_students]
if "sidebar_students_multiselect" in st.session_state:
    cur_labels = set(st.session_state["sidebar_students_multiselect"])
    st.session_state["sidebar_students_multiselect"] = sorted(cur_labels.union(labels_for_selected))

# Multiselect with colored-label options
selected_opts = st.sidebar.multiselect(
    "Students (toggle visibility)",
    options=list(options_map.keys()),
    default=labels_for_selected,
    key="sidebar_students_multiselect",
)

# Map back to plain names
st.session_state.selected_students = [options_map[o] for o in selected_opts]

# =========================
# Main Tabs
# =========================

tab_results, tab_transcripts = st.tabs(["📈 Results", "💬 Transcripts"])

# =========================
# Results Tab — sequential sections per experiment
# =========================

with tab_results:
    st.subheader("Aggregate Results (by experiment)")
    st.caption(f"Teacher: `{teacher}` · Dataset: `{dataset}`")

    any_plots = False
    for exp in all_experiments:
        exp_dir = str(Path(exp_root) / exp)
        lay = discover_layout(exp_dir, layout_signature(Path(exp_dir)))  # name-aware layout cache key
        if teacher not in lay:
            continue
        # Students present in this experiment for chosen dataset
        exp_students = sorted([s for s, ds in lay[teacher].items() if dataset in ds])
        # Apply multiselect filter
        active_students = [s for s in exp_students if s in st.session_state.selected_students]
        if not active_students:
            continue

        df = build_results_df(exp_dir, teacher, dataset, active_students)
        # Silently skip experiments that have no data for the current selection
        if df.empty:
            continue

        st.markdown(f"### `{exp}`")

        with st.expander("Show data table", expanded=False):
            st.dataframe(df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        title_suffix = f"({teacher}, {dataset})"
        with col1:
            st.altair_chart(plot_passk(df, 1, student_colors, title_suffix), use_container_width=True)
        with col2:
            st.altair_chart(plot_passk(df, 5, student_colors, title_suffix), use_container_width=True)
        with col3:
            st.altair_chart(plot_passk(df, 10, student_colors, title_suffix), use_container_width=True)

        any_plots = True
        st.markdown("---")

    if not any_plots:
        st.info("No matching results across experiments for the current selection and students.")

# =========================
# Transcripts Tab — experiment select inside the tab
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
                        st.markdown(f"> {question}")
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
