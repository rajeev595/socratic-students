"""
Model Interaction Dashboard (Streamlit)

A lightweight viewer for:
- Aggregated pass@k results across turns (with error bands)
- Per-problem transcripts rendered as a chat UI

Directory layout (auto-discovered):
    EXP_DIR/
      TEACHER/
        STUDENT/
          DATASET/
            aggregate_results.json
            transcripts/
              <pid>.json

Caching strategy:
- We compute inexpensive directory/file "signatures" (count/mtime or size/mtime)
  and include them in st.cache_data keys so the UI auto-refreshes whenever
  files are added/changed/removed—without forcing a full app rerun.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import (
    Dict, List, Tuple, Optional, NamedTuple, TypedDict, Iterable, Literal
)

import pandas as pd
import streamlit as st
import altair as alt

# =========================
# Config & Utilities
# =========================

st.set_page_config(page_title="Model Interaction Dashboard", layout="wide")

# ---- Types ----

class FileSig(NamedTuple):
    """Immutable file signature used to invalidate caches."""
    exists: bool
    size: int
    mtime: float

# Mapping: teacher -> (student -> [dataset])
LayoutType = Dict[str, Dict[str, List[str]]]

class TranscriptEntry(TypedDict, total=False):
    """Loose schema for a single transcript line."""
    role: str          # e.g., "openai/gpt-4o", "meta/llama3-8b", etc.
    turn: int          # numeric turn index if present
    content: str       # message content


def _is_dir(p: Path) -> bool:
    """Return True if path is a directory, ignoring permission/IO errors."""
    try:
        return p.is_dir()
    except Exception:
        return False


def file_signature(p: Path) -> FileSig:
    """
    Return a stable signature for a single file.

    We include (exists, size, mtime). Any change causes a cache miss for
    functions keyed with this signature.
    """
    try:
        stt = p.stat()
        return FileSig(True, stt.st_size, stt.st_mtime)
    except Exception:
        return FileSig(False, 0, 0.0)


def dir_signature(path: Path) -> Tuple[int, float]:
    """
    Return (items_count, latest_mtime) for an entire directory tree.

    Any added/removed/modified file updates this signature.
    For non-existent paths, returns (0, 0.0).
    """
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
    """JSON loader that returns None on failure instead of raising."""
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# =========================
# Cached loaders (keys include signatures)
# =========================

@st.cache_data(show_spinner=False)
def discover_layout(exp_dir: str, sig: Tuple[int, float]) -> LayoutType:
    """
    Discover teachers, students, datasets under EXP_DIR.

    Args:
        exp_dir: Root experiments directory.
        sig: Signature of the root dir (items_count, latest_mtime). This is only
             used to bust the cache when the FS changes.

    Returns:
        {teacher: {student: [dataset, ...], ...}, ...}
    """
    base = Path(exp_dir)
    layout: LayoutType = {}
    if not _is_dir(base):
        return layout

    for teacher_dir in sorted([p for p in base.iterdir() if _is_dir(p)]):
        teacher = teacher_dir.name
        layout[teacher] = {}
        for student_dir in sorted([p for p in teacher_dir.iterdir() if _is_dir(p)]):
            student = student_dir.name
            datasets: List[str] = []
            for dataset_dir in sorted([p for p in student_dir.iterdir() if _is_dir(p)]):
                # A dataset is considered valid if it has either results or transcripts
                has_results = (dataset_dir / "aggregate_results.json").exists()
                has_trans   = _is_dir(dataset_dir / "transcripts")
                if has_results or has_trans:
                    datasets.append(dataset_dir.name)
            if datasets:
                layout[teacher][student] = datasets
        if not layout[teacher]:
            # No valid dataset under this teacher—drop it
            del layout[teacher]
    return layout


@st.cache_data(show_spinner=False)
def load_aggregate_results(
    exp_dir: str,
    teacher: str,
    student: str,
    dataset: str,
    sig: FileSig
) -> Optional[dict]:
    """
    Load aggregate_results.json with a file signature for cache invalidation.

    Args:
        sig: Build with file_signature(path-to-aggregate_results.json).
    """
    p = Path(exp_dir) / teacher / student / dataset / "aggregate_results.json"
    return _load_json(p)


@st.cache_data(show_spinner=False)
def list_pids(
    exp_dir: str,
    teacher: str,
    student: str,
    dataset: str,
    sig: Tuple[int, float]
) -> List[str]:
    """
    List transcript PIDs (filenames without .json) for a dataset/student.

    Args:
        sig: dir_signature(transcripts_dir(...)) so new/removed files appear.
    """
    tdir = transcripts_dir(exp_dir, teacher, student, dataset)
    if not _is_dir(tdir):
        return []
    pids = [p.stem for p in tdir.glob("*.json")]
    # Sort numerically when possible; otherwise lexicographically.
    pids = sorted(pids, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s))
    return pids


@st.cache_data(show_spinner=False)
def load_transcript(
    exp_dir: str,
    teacher: str,
    student: str,
    dataset: str,
    pid: str,
    sig: FileSig
) -> Optional[List[TranscriptEntry]]:
    """
    Load a single transcript file (list of entries).

    Args:
        sig: file_signature(path-to-<pid>.json) so edits auto-refresh.

    Returns:
        List of transcript dicts (role/turn/content), or None on failure.
    """
    p = transcripts_dir(exp_dir, teacher, student, dataset) / f"{pid}.json"
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else None
    except Exception:
        return None


# =========================
# Data shaping & plotting
# =========================

def agg_to_long_df(agg: dict, student: str) -> pd.DataFrame:
    """
    Convert aggregate_results.json into a tidy long-form DataFrame.

    Input shape (example):
        {
          "1": {"1": [mean, std], "5": [mean, std], "10": [mean, std]},
          "3": {...},
          ...
        }

    Output columns:
        ['turn', 'k', 'mean', 'std', 'student']
    """
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
    """
    Concatenate aggregate results across selected students for a (teacher, dataset).

    Uses file signatures per student to invalidate caches when JSONs change.
    """
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


def plot_passk(df: pd.DataFrame, k_val: int, title_suffix: str = "") -> alt.Chart:
    """
    Altair line chart with ±1 std error band for a given k (1, 5, 10).
    """
    sub = df[df["k"] == k_val].copy()
    if sub.empty:
        # Return an empty chart with a title so the layout stays consistent
        return alt.Chart(pd.DataFrame({"turn": [], "mean": [], "student": []})).mark_line().properties(
            title=f"pass@{k_val} {title_suffix}".strip()
        )

    band = (
        alt.Chart(sub)
        .transform_calculate(upper="datum.mean + datum.std", lower="datum.mean - datum.std")
        .mark_area(opacity=0.2)
        .encode(
            x=alt.X("turn:Q", title="Turn"),
            y=alt.Y("lower:Q", title=f"pass@{k_val}"),
            y2="upper:Q",
            color=alt.Color("student:N", legend=None),
        )
    )

    lines = (
        alt.Chart(sub)
        .mark_line(point=True)
        .encode(
            x=alt.X("turn:Q", title="Turn"),
            y=alt.Y("mean:Q", title=f"pass@{k_val}"),
            color=alt.Color("student:N", title="Student"),
            tooltip=[
                "student",
                "turn",
                alt.Tooltip("mean:Q", format=".3f"),
                alt.Tooltip("std:Q", format=".3f"),
            ],
        )
    )

    return (band + lines).properties(title=f"pass@{k_val} {title_suffix}".strip(), height=300).interactive()


def transcripts_dir(exp_dir: str, teacher: str, student: str, dataset: str) -> Path:
    """Helper to build the transcripts/ directory path."""
    return Path(exp_dir) / teacher / student / dataset / "transcripts"


def _norm(s: str) -> str:
    """
    Normalize model 'role' strings so we can heuristically match to teacher/student.

    Examples:
        "openai/gpt-4o" -> "gpt4o"
        "meta/llama-3-8b" -> "llama38b"
    """
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    for tok in ["openai/", "meta/", "google/", "qwen/"]:
        s = s.replace(tok, "")
    for ch in ["/", "_", "-", " "]:
        s = s.replace(ch, "")
    return s


def model_role_for_chat(
    entry_role: str,
    turn: Optional[int],
    teacher: str,
    student: str
) -> Tuple[Literal["teacher", "student"], str, str]:
    """
    Decide whether a transcript row should be rendered as teacher or student.

    Heuristics:
    1) Exact normalized match to teacher/student name if present.
    2) If turn number exists: even->teacher, odd->student.
    3) Fallback: substring containment.

    Returns:
        (who, display_name, emoji)
            who: "teacher" | "student"
            display_name: original role if available, otherwise teacher/student name
            emoji: 🧑‍🏫 for teacher, 🧑‍🎓 for student
    """
    er = _norm(entry_role)
    t  = _norm(teacher)
    s  = _norm(student)

    if er and er == t:
        return "teacher", entry_role or teacher, "🧑‍🏫"
    if er and er == s:
        return "student", entry_role or student, "🧑‍🎓"

    if isinstance(turn, int):
        if turn % 2 == 0:
            return "teacher", entry_role or teacher, "🧑‍🏫"
        else:
            return "student", entry_role or student, "🧑‍🎓"

    is_teacher = bool(er and t and (er in t or t in er))
    is_student = bool(er and s and (er in s or s in er))
    if is_teacher and not is_student:
        return "teacher", entry_role or teacher, "🧑‍🏫"
    if is_student and not is_teacher:
        return "student", entry_role or student, "🧑‍🎓"
    return "teacher", entry_role or teacher, "🧑‍🏫"


# =========================
# Sidebar (Global Controls)
# =========================

st.sidebar.title("Controls")

# Root experiments folder (fixed by design; use a text_input if you want it editable)
DEFAULT_EXP_DIR = "/playpen-ssd/ambati/interact_plus/experiments/test/"
# If you want this to be user-editable, replace the next line with:
# exp_dir = st.sidebar.text_input("Experiments Directory", DEFAULT_EXP_DIR)
exp_dir: str = DEFAULT_EXP_DIR

# Discover layout with root dir signature so new folders appear automatically
root_sig = dir_signature(Path(exp_dir))
layout = discover_layout(exp_dir, root_sig)

if not layout:
    st.sidebar.warning("No teachers/students/datasets found under the given directory.")
    st.title("Model Interaction Dashboard")
    st.write(
        "Set the correct **Experiments Directory** in the sidebar "
        "or ensure your folder structure matches the expected layout."
    )
    st.stop()

# Teacher selection
all_teachers = sorted(layout.keys())
teacher = st.sidebar.selectbox("Teacher", all_teachers)

# Dataset selection (union over students for chosen teacher)
datasets_for_teacher = sorted({d for s, ds in layout[teacher].items() for d in ds})
if not datasets_for_teacher:
    st.sidebar.error("No datasets found for the selected teacher.")
    st.stop()

dataset = st.sidebar.selectbox("Dataset", datasets_for_teacher)

# Students under this teacher that have the chosen dataset
students_for_dataset = sorted([s for s, ds in layout[teacher].items() if dataset in ds])
if not students_for_dataset:
    st.sidebar.error("No students found for the chosen teacher/dataset.")
    st.stop()

# For results: allow toggling which students to include (default: all)
selected_students = st.sidebar.multiselect(
    "Students (for plots)",
    students_for_dataset,
    default=students_for_dataset
)

# Helpful description for users
st.sidebar.caption(
    "This app auto-detects new results/transcripts via directory signatures. "
    "Drop files in the expected folders and the views will refresh."
)

# =========================
# Main Tabs
# =========================

tab_results, tab_transcripts = st.tabs(["📈 Results", "💬 Transcripts"])

# =========================
# Results Tab
# =========================

with tab_results:
    st.subheader("Aggregate Results")
    st.caption(f"Teacher: `{teacher}` · Dataset: `{dataset}`")

    df = build_results_df(exp_dir, teacher, dataset, selected_students)
    if df.empty:
        st.info("No aggregate_results.json found for the current selection.")
    else:
        with st.expander("Show data table"):
            st.dataframe(df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.altair_chart(plot_passk(df, 1, f"({teacher}, {dataset})"), use_container_width=True)
        with col2:
            st.altair_chart(plot_passk(df, 5, f"({teacher}, {dataset})"), use_container_width=True)
        with col3:
            st.altair_chart(plot_passk(df, 10, f"({teacher}, {dataset})"), use_container_width=True)

# =========================
# Transcripts Tab
# =========================

with tab_transcripts:
    st.subheader("Conversation Viewer")
    st.caption("Render a transcript as a chat between the selected teacher and student.")

    # Student selector for the chat viewer (independent from plots)
    t_student = st.selectbox("Student (for transcript)", students_for_dataset, index=0)

    # Use transcripts dir signature so new/removed transcript files show up
    tdir = transcripts_dir(exp_dir, teacher, t_student, dataset)
    t_sig = dir_signature(tdir)
    pids = list_pids(exp_dir, teacher, t_student, dataset, t_sig)

    if not pids:
        st.info("No transcripts found (no JSON files in transcripts/).")
    else:
        pid = st.selectbox("Problem ID (pid)", pids, index=0)

        # Per-file signature to auto-refresh when the chosen transcript changes
        this_file = tdir / f"{pid}.json"
        f_sig = file_signature(this_file)
        data = load_transcript(exp_dir, teacher, t_student, dataset, pid, f_sig)
        if not isinstance(data, list):
            st.warning("Transcript JSON could not be loaded or has unexpected format.")
        else:
            st.write(
                f"**Teacher:** `{teacher}` · **Student:** `{t_student}` · "
                f"**Dataset:** `{dataset}` · **pid:** `{pid}`"
            )

            # Inject minimal CSS for chat bubbles
            st.markdown(
                """
                <style>
                .chat-row { display: flex; width: 100%; margin: 0.10rem 0; }
                .chat-left  { justify-content: flex-start; }
                .chat-right { justify-content: flex-end;   }

                .bubble-wrap {
                    display: flex; align-items: flex-start; gap: 0.4rem; max-width: 70%;
                }
                .wrap-left  { flex-direction: row; }
                .wrap-right { flex-direction: row-reverse; }

                .avatar-emoji { font-size: 3.6rem; line-height: 1; margin-top: 2px; }
                .wrap-left  .avatar { margin-right: 0.6rem; }
                .wrap-right .avatar { margin-left: 0.6rem; }

                .bubble {
                    padding: 0.4rem 0.7rem; border-radius: 12px; line-height: 1.35;
                    border: 1px solid rgba(0,0,0,0.06); box-shadow: 0 1px 2px rgba(0,0,0,0.04);
                    word-wrap: break-word; white-space: pre-wrap; font-size: 1rem;
                }
                .bubble-teacher { background: #f1f5ff; }
                .bubble-student { background: #f7f7f7; }

                .meta { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.15rem; }
                .name { font-weight: 700; }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Render each transcript line as a bubble
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
