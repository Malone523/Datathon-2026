"""
Process Influence Explorer
──────────────────────────
Table 1: Interactive counterfactual simulator
  • Pick variable → see current mean + direction
  • Type your adjustment → see predicted scrap Δ live
Table 2: Machine performance + top OLS driver
"""

import json
import threading
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
import statsmodels.api as sm
from scipy import stats
import lingam


# ══════════════════════════════════════════════════════
#  DATA & ONTOLOGY
# ══════════════════════════════════════════════════════

with open("injection_molding_ontology.json") as f:
    ONTOLOGY = json.load(f)

COL_META = {m["name"]: m for m in ONTOLOGY["column_metadata"]}
df_flat  = pd.read_csv("synthetic_injection_molding_demo.csv")

SCRAP_TARGET  = "scrap_rate_pct"
SCRAP_PARENTS = [
    c for c in COL_META[SCRAP_TARGET]["likely_causal_parents"]
    if c in df_flat.select_dtypes(include=[np.number]).columns
]

_EXCLUDED_ROLES = {"identifier", "target", "target_support", "control"}
CONTROLLABLE = [
    m for m in ONTOLOGY["column_metadata"]
    if m["suitable_for_counterfactual_intervention"]
    and m["role"] not in _EXCLUDED_ROLES
    and m["name"] in df_flat.columns
]
LABEL_TO_COL = {m["human_readable_label"]: m["name"] for m in CONTROLLABLE}
COL_TO_LABEL = {m["name"]: m["human_readable_label"] for m in CONTROLLABLE}
VAR_LABELS   = sorted(LABEL_TO_COL.keys())


# ══════════════════════════════════════════════════════
#  GLOBAL OLS  (backdoor-adjusted, direct parents only)
# ══════════════════════════════════════════════════════

_vg     = df_flat[SCRAP_PARENTS + [SCRAP_TARGET]].dropna()
_Xg     = sm.add_constant(_vg[SCRAP_PARENTS])
OLS_G   = sm.OLS(_vg[SCRAP_TARGET], _Xg).fit()

_OLS_COEF_CACHE: dict[str, float] = {col: OLS_G.params[col] for col in SCRAP_PARENTS}
_OLS_PVAL_CACHE: dict[str, float] = {col: OLS_G.pvalues[col] for col in SCRAP_PARENTS}


def get_coef(col_name: str) -> tuple[float, float, bool]:
    if col_name in _OLS_COEF_CACHE:
        return _OLS_COEF_CACHE[col_name], _OLS_PVAL_CACHE[col_name], True
    if col_name not in df_flat.columns:
        return 0.0, 1.0, False
    valid = df_flat[[col_name, SCRAP_TARGET]].dropna()
    if valid.empty or valid[col_name].std() == 0:
        return 0.0, 1.0, False
    Xb  = sm.add_constant(valid[col_name])
    res = sm.OLS(valid[SCRAP_TARGET], Xb).fit()
    c, p = float(res.params[col_name]), float(res.pvalues[col_name])
    _OLS_COEF_CACHE[col_name] = c
    _OLS_PVAL_CACHE[col_name] = p
    return c, p, False


# ══════════════════════════════════════════════════════
#  LiNGAM  (background, 800-row subsample)
# ══════════════════════════════════════════════════════

LINGAM_RESULTS: dict = {}
LINGAM_READY   = threading.Event()

def _run_lingam():
    try:
        cols   = SCRAP_PARENTS + [SCRAP_TARGET]
        data   = df_flat[cols].dropna()
        sample = data.sample(n=min(800, len(data)), random_state=42)
        std    = (sample - sample.mean()) / sample.std()
        mdl    = lingam.DirectLiNGAM()
        mdl.fit(std)
        idx_s  = cols.index(SCRAP_TARGET)
        order  = [cols[i] for i in mdl.causal_order_]
        s_pos  = order.index(SCRAP_TARGET) if SCRAP_TARGET in order else len(order)
        LINGAM_RESULTS["effects"] = {
            cols[i]: float(mdl.adjacency_matrix_[idx_s, i])
            for i in range(len(cols))
            if cols[i] in order and order.index(cols[i]) < s_pos
            and abs(mdl.adjacency_matrix_[idx_s, i]) > 0.001
        }
        LINGAM_RESULTS["order"] = order
    except Exception as e:
        LINGAM_RESULTS["error"] = str(e)
    finally:
        LINGAM_READY.set()

threading.Thread(target=_run_lingam, daemon=True).start()


# ══════════════════════════════════════════════════════
#  MACHINE SCORING  (Table 2)
# ══════════════════════════════════════════════════════

_agg = df_flat.groupby("machine_id").agg(
    produced   = ("parts_produced",     "sum"),
    scrap      = ("scrap_count",         "sum"),
    energy_kwh = ("energy_kwh_interval", "sum"),
    cycle_s    = ("cycle_time_s",        "mean"),
).reset_index()

PRODUCTION_LINES = {
    r["machine_id"]: {
        "produced": int(r["produced"]), "scrap": int(r["scrap"]),
        "energy_kwh": float(r["energy_kwh"]), "cycle_s": float(r["cycle_s"]),
    }
    for _, r in _agg.iterrows()
}
LINE_OPTIONS = sorted(PRODUCTION_LINES.keys())

_MACHINE_OLS: dict = {}

def get_machine_ols(mid: str):
    if mid not in _MACHINE_OLS:
        sub = df_flat[df_flat["machine_id"] == mid][SCRAP_PARENTS + [SCRAP_TARGET]].dropna()
        try:
            _MACHINE_OLS[mid] = sm.OLS(sub[SCRAP_TARGET], sm.add_constant(sub[SCRAP_PARENTS])).fit() if len(sub) >= 30 else None
        except Exception:
            _MACHINE_OLS[mid] = None
    return _MACHINE_OLS[mid]

SCORE_W = {"scrap": 0.5, "energy": 0.3, "cycle": 0.2}

def problem_score(d):
    p = d["produced"]
    if p == 0: return float("inf")
    return SCORE_W["scrap"]*(d["scrap"]/p) + SCORE_W["energy"]*(d["energy_kwh"]/p) + SCORE_W["cycle"]*(d["cycle_s"]/60)

_scores    = sorted(s for s in (problem_score(v) for v in PRODUCTION_LINES.values()) if s != float("inf"))
_sc_min    = _scores[0]
_sc_range  = max(_scores[-1] - _sc_min, 1e-9)
_p33, _p67 = float(np.percentile(_scores, 33)), float(np.percentile(_scores, 67))

def score_col(s):
    if s == float("inf"): return "#ff4444"
    if s > _p67: return "#ff8c00"
    if s > _p33: return "#e6c800"
    return "#4caf50"

def score_pct(s):  return float(np.clip((s - _sc_min) / _sc_range, 0, 1))
def score_rank(s): return f"rank {sum(1 for x in _scores if x <= s)} / {len(_scores)}"

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

_CAUSAL_CONTEXT = {
    "ambient_humidity_pct":        "Humidity → dryer saturation → resin moisture → splay → scrap",
    "dryer_dewpoint_c":            "Humidity → dryer dew point → resin moisture → splay → scrap",
    "resin_moisture_pct":          "Humidity → dryer → resin moisture → moisture defects → scrap",
    "mold_temperature_c":          "Mold temp + cooling time → warpage risk → scrap",
    "cooling_time_s":              "Mold temp + cooling time → warpage risk → scrap",
    "maintenance_days_since_last": "Maintenance gap → calibration drift → instability → scrap",
    "calibration_drift_index":     "Maintenance → calibration drift → process instability → scrap",
    "tool_wear_index":             "Tool wear + injection pressure + clamp force → flash → scrap",
    "injection_pressure_bar":      "Tool wear + pressure + clamp force → flash / short shot → scrap",
    "barrel_temperature_c":        "Batch quality + barrel temp + pressure → short shot → scrap",
    "hold_pressure_bar":           "Hold pressure → sink mark / dimensional deviation → scrap",
    "screw_speed_rpm":             "Screw speed → shear heat → burn mark → scrap",
    "shot_size_g":                 "Shot size → flash / short shot → scrap",
}

def machine_top_driver(mid: str):
    ols_m = get_machine_ols(mid)
    if ols_m:
        sub = df_flat[df_flat["machine_id"] == mid]
        best_col, best_std, best_p = None, 0.0, 1.0
        for col in SCRAP_PARENTS:
            if col not in ols_m.params: continue
            s = abs(ols_m.params[col] * sub[col].std())
            if s > best_std:
                best_col, best_std, best_p = col, s, ols_m.pvalues[col]
        if best_col:
            label = COL_TO_LABEL.get(best_col, best_col)
            unit  = COL_META.get(best_col, {}).get("unit") or ""
            disp  = f"{label} ({unit})" if unit else label
            coef  = ols_m.params[best_col]
            annot = (f"OLS coef = {coef:+.4f}  {sig_stars(best_p)}\n"
                     f"Std effect = {best_std:+.3f} % / SD\n"
                     f"Machine R² = {ols_m.rsquared:.3f}")
            return best_col, disp, annot
    return "—", "No data", "—"


# ══════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════

BG        = "#1a1a1a"
CELL_BG   = "#252525"
HEADER_BG = "#2e2e2e"
CARD_BG   = "#1f1f1f"
BORDER    = "#383838"
FG        = "#e8e8e8"
DIM_FG    = "#555555"
MID_FG    = "#888888"
GREEN     = "#4caf50"
RED       = "#f44336"
YELLOW    = "#e6c800"
FONT      = ("Consolas", 11)
FONT_SM   = ("Consolas", 9)
FONT_LBL  = ("Consolas", 10)
FONT_BIG  = ("Consolas", 15, "bold")
FONT_HUGE = ("Consolas", 22, "bold")


# ══════════════════════════════════════════════════════
#  ROOT WINDOW
# ══════════════════════════════════════════════════════

root = tk.Tk()
root.title("Process Influence Explorer")
root.geometry("1100x720")
root.configure(bg=BG)
root.resizable(True, True)

sty = ttk.Style()
sty.theme_use("default")
sty.configure("Dark.TCombobox",
    fieldbackground=CELL_BG, background=CELL_BG,
    foreground=FG, arrowcolor=FG, font=FONT)
sty.map("Dark.TCombobox",
    fieldbackground=[("readonly", CELL_BG)],
    foreground=[("readonly", FG)])


# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════

def section_label(text: str, pady_top: int = 16):
    tk.Label(root, text=text, font=FONT_LBL,
             bg=BG, fg=DIM_FG).pack(anchor="w", padx=20, pady=(pady_top, 4))

def make_header_cell(parent, text, col, span=1):
    lbl = tk.Label(parent, text=text, font=FONT_SM,
                   bg=HEADER_BG, fg=MID_FG, anchor="w",
                   padx=8, pady=6)
    lbl.grid(row=0, column=col, columnspan=span,
             padx=2, pady=(2, 0), sticky="ew")
    return lbl

def make_text_cell(parent, height, row, col, colspan=1):
    t = tk.Text(parent, height=height,
                font=FONT, bg=CELL_BG, fg=FG,
                wrap=tk.WORD, relief="flat",
                state="disabled", cursor="arrow",
                padx=5, pady=5,
                selectbackground=CELL_BG,
                inactiveselectbackground=CELL_BG,
                highlightthickness=1,
                highlightbackground=BORDER,
                highlightcolor=BORDER)
    t.grid(row=row, column=col, columnspan=colspan,
           padx=2, pady=2, sticky="nsew")
    return t

def fill_text(w, text, fg=FG):
    w.config(state="normal", fg=fg)
    w.delete("1.0", tk.END)
    w.insert("1.0", text)
    w.config(state="disabled")


# ══════════════════════════════════════════════════════
#  TABLE 1 — COUNTERFACTUAL SIMULATOR
# ══════════════════════════════════════════════════════

section_label("Counterfactual Simulator  ·  pick a variable → type your adjustment → see scrap impact")

sim_frame = tk.Frame(root, bg=BG)
sim_frame.pack(padx=20, fill=tk.X)

# ── uniform column config: all 5 columns equal weight ─────────────────────
for ci in range(5):
    sim_frame.columnconfigure(ci, weight=1, minsize=140)

# ── headers: no fixed width — grid/weight handles sizing ──────────────────
for ci, txt in enumerate([
    "Variable",
    "Current avg",
    "Effect on scrap",
    "Your adjustment",
    "Predicted scrap Δ",
]):
    make_header_cell(sim_frame, txt, ci)

# ── row 1, col 0: variable dropdown ───────────────────────────────────────

var_cb = ttk.Combobox(sim_frame, values=VAR_LABELS,
    style="Dark.TCombobox", state="readonly", font=FONT)
var_cb.grid(row=1, column=0, padx=2, pady=4, sticky="ew", ipady=4)

# ── row 1, col 1: current average + unit ──────────────────────────────────

cur_val_lbl = tk.Label(sim_frame, text="—",
    font=FONT, bg=CELL_BG, fg=FG,
    anchor="w", padx=8, pady=8,
    highlightthickness=1, highlightbackground=BORDER)
cur_val_lbl.grid(row=1, column=1, padx=2, pady=4, sticky="ew")

# ── row 1, col 2: direction badge ─────────────────────────────────────────

dir_badge = tk.Label(sim_frame, text="—",
    font=FONT, bg=CELL_BG, fg=MID_FG,
    anchor="center", padx=8, pady=8,
    highlightthickness=1, highlightbackground=BORDER)
dir_badge.grid(row=1, column=2, padx=2, pady=4, sticky="ew")

# ── row 1, col 3: adjustment entry + constraint labels ────────────────────

adj_outer = tk.Frame(sim_frame, bg=BG)
adj_outer.grid(row=1, column=3, padx=2, pady=4, sticky="ew")
adj_outer.columnconfigure(0, weight=1)

adj_entry = tk.Entry(adj_outer,
    font=FONT_BIG, bg=CELL_BG, fg=FG,
    insertbackground=FG, relief="flat",
    highlightthickness=1,
    highlightbackground=BORDER,
    highlightcolor="#666",
    justify="center")
adj_entry.grid(row=0, column=0, sticky="ew", ipady=8)

adj_hint = tk.Label(adj_outer,
    text="e.g.  +2.5  or  -10",
    font=FONT_SM, bg=BG, fg=DIM_FG)
adj_hint.grid(row=1, column=0, sticky="w", pady=(2, 0))

adj_range_lbl = tk.Label(adj_outer,
    text="",
    font=FONT_SM, bg=BG, fg=DIM_FG)
adj_range_lbl.grid(row=2, column=0, sticky="w")

# ── row 1, col 4: result card ─────────────────────────────────────────────

result_card = tk.Frame(sim_frame, bg=CARD_BG,
    highlightthickness=1, highlightbackground=BORDER)
result_card.grid(row=1, column=4, padx=2, pady=4, sticky="nsew")
result_card.columnconfigure(0, weight=1)

result_delta_lbl = tk.Label(result_card,
    text="—", font=FONT_HUGE,
    bg=CARD_BG, fg=MID_FG, anchor="center")
result_delta_lbl.pack(fill=tk.X, padx=12, pady=(14, 2))

result_abs_lbl = tk.Label(result_card,
    text="", font=FONT_SM,
    bg=CARD_BG, fg=MID_FG, anchor="center")
result_abs_lbl.pack(fill=tk.X, padx=12, pady=(0, 4))

result_method_lbl = tk.Label(result_card,
    text="", font=FONT_SM,
    bg=CARD_BG, fg=DIM_FG, anchor="center")
result_method_lbl.pack(fill=tk.X, padx=12, pady=(0, 10))

# ── causal note below (spans full width) ──────────────────────────────────

causal_note = tk.Label(sim_frame,
    text="",
    font=FONT_SM, bg=BG, fg=DIM_FG,
    anchor="w", wraplength=900, justify="left")
causal_note.grid(row=2, column=0, columnspan=5,
                 padx=4, pady=(0, 6), sticky="w")


# ── state tracking ────────────────────────────────────────────────────────

_sim_state = {
    "col_name": None,
    "coef": 0.0,
    "p_val": 1.0,
    "is_direct": False,
    "mean": 0.0,
    "spec_min": None,
    "spec_max": None,
}


def _recalc(*_):
    col  = _sim_state["col_name"]
    coef = _sim_state["coef"]
    p    = _sim_state["p_val"]
    mean = _sim_state["mean"]

    raw = adj_entry.get().strip()
    if not col or not raw:
        result_delta_lbl.config(text="—", fg=MID_FG)
        result_abs_lbl.config(text="")
        result_method_lbl.config(text="")
        return

    try:
        delta = float(raw)
    except ValueError:
        result_delta_lbl.config(text="?", fg=YELLOW)
        result_abs_lbl.config(text="enter a number")
        return

    spec_min = _sim_state["spec_min"]
    spec_max = _sim_state["spec_max"]
    if spec_min is not None and spec_max is not None:
        new_val  = float(np.clip(mean + delta, spec_min, spec_max))
        eff_delta = new_val - mean
    else:
        eff_delta = delta

    scrap_delta = coef * eff_delta

    sign   = "▼" if scrap_delta < 0 else "▲"
    colour = GREEN if scrap_delta < 0 else RED
    mean_scrap = float(df_flat[SCRAP_TARGET].mean())

    result_delta_lbl.config(
        text=f"{sign} {abs(scrap_delta):.3f} %",
        fg=colour)

    new_scrap = mean_scrap + scrap_delta
    result_abs_lbl.config(
        text=f"avg scrap:  {mean_scrap:.2f} %  →  {new_scrap:.2f} %",
        fg=MID_FG)

    stars = sig_stars(p)
    direct_str = "direct causal" if _sim_state["is_direct"] else "total (indirect)"
    result_method_lbl.config(
        text=f"OLS {direct_str} coef = {coef:+.4f}  {stars}",
        fg=DIM_FG)


adj_entry.bind("<KeyRelease>", _recalc)


def on_var_selected(event=None):
    label    = var_cb.get()
    col_name = LABEL_TO_COL.get(label)
    if not col_name:
        return

    coef, p_val, is_direct = get_coef(col_name)
    meta    = COL_META.get(col_name, {})
    unit    = meta.get("unit") or ""
    unit_s  = f" {unit}" if unit else ""
    allowed = meta.get("allowed_range_or_categories")
    col_data = df_flat[col_name].dropna()
    mean_val = float(col_data.mean())
    spec_min = allowed[0] if isinstance(allowed, list) and len(allowed) == 2 else None
    spec_max = allowed[1] if isinstance(allowed, list) and len(allowed) == 2 else None

    _sim_state.update({
        "col_name": col_name, "coef": coef, "p_val": p_val,
        "is_direct": is_direct, "mean": mean_val,
        "spec_min": spec_min, "spec_max": spec_max,
    })

    cur_val_lbl.config(text=f"{mean_val:.2f}{unit_s}")

    if p_val >= 0.05:
        dir_badge.config(
            text=f"~ neutral\n(p={p_val:.2f} ns)",
            fg=MID_FG, bg=CELL_BG)
    elif coef > 0:
        dir_badge.config(
            text=f"↑ raises scrap\n{sig_stars(p_val)}  p={p_val:.4f}",
            fg=RED, bg=CELL_BG)
    else:
        dir_badge.config(
            text=f"↓ lowers scrap\n{sig_stars(p_val)}  p={p_val:.4f}",
            fg=GREEN, bg=CELL_BG)

    if spec_min is not None:
        adj_range_lbl.config(
            text=f"spec: {spec_min}–{spec_max}{unit_s}  |  mean: {mean_val:.2f}{unit_s}")
    else:
        obs_min, obs_max = col_data.min(), col_data.max()
        adj_range_lbl.config(
            text=f"observed: {obs_min:.1f}–{obs_max:.1f}{unit_s}  |  mean: {mean_val:.2f}{unit_s}")

    pathway = _CAUSAL_CONTEXT.get(col_name, "")
    direct_note = ("  ·  backdoor-adjusted OLS (direct causal effect)"
                   if is_direct else
                   "  ·  bivariate OLS — total/indirect effect, not fully adjusted")
    lingam_part = ""
    if LINGAM_READY.is_set() and "effects" in LINGAM_RESULTS:
        fx = LINGAM_RESULTS["effects"].get(col_name)
        if fx is not None:
            agree = (fx > 0) == (coef > 0)
            lingam_part = f"  |  LiNGAM: {fx:+.3f} ({'✓ confirms' if agree else '⚠ sign differs'})"
    causal_note.config(text=pathway + direct_note + lingam_part)

    adj_entry.delete(0, tk.END)
    result_delta_lbl.config(text="—", fg=MID_FG)
    result_abs_lbl.config(text="")
    result_method_lbl.config(text="")
    adj_entry.focus_set()


var_cb.bind("<<ComboboxSelected>>", on_var_selected)


# ══════════════════════════════════════════════════════
#  TABLE 2 — MACHINE PERFORMANCE
# ══════════════════════════════════════════════════════

section_label("Machine Performance  ·  composite score  |  machine-level OLS top driver  →",
              pady_top=22)

frame2 = tk.Frame(root, bg=BG)
frame2.pack(padx=20, fill=tk.X)

# ── uniform column config matching Table 1's 5-col total width ────────────
# 4 cols: Machine(1) | Score(1) | Top Driver(2) | Causal Pathway(2) — weight sum = 6
# Table 1 has 5 cols all weight=1 — total weight=5
# Use weights [1, 1, 2, 2] so both tables feel proportionally balanced
for ci, weight in enumerate([1, 1, 2, 2]):
    frame2.columnconfigure(ci, weight=weight, minsize=140)

T2_HEADERS = ["Machine", "Problem Score", "Top Scrap Driver (OLS)", "Causal Pathway"]

for ci, txt in enumerate(T2_HEADERS):
    make_header_cell(frame2, txt, ci)

line_cb = ttk.Combobox(frame2, values=LINE_OPTIONS,
    style="Dark.TCombobox", state="readonly", font=FONT)
line_cb.grid(row=1, column=0, padx=2, pady=2, sticky="new", ipady=4)

score_card = tk.Frame(frame2, bg=CARD_BG,
    highlightthickness=1, highlightbackground=BORDER)
score_card.grid(row=1, column=1, padx=2, pady=2, sticky="nsew")
score_card.columnconfigure(0, weight=1)

score_num_lbl = tk.Label(score_card, text="—",
    font=FONT_BIG, bg=CARD_BG, fg=FG, anchor="w")
score_num_lbl.pack(fill=tk.X, padx=10, pady=(12, 2))

score_bar_cv = tk.Canvas(score_card, height=5,
    bg=CARD_BG, highlightthickness=0)
score_bar_cv.pack(fill=tk.X, padx=10, pady=(2, 4))

score_rank_lbl = tk.Label(score_card, text="",
    font=FONT_SM, bg=CARD_BG, fg=MID_FG, anchor="w")
score_rank_lbl.pack(fill=tk.X, padx=10, pady=(0, 4))

score_r2_lbl = tk.Label(score_card, text="",
    font=FONT_SM, bg=CARD_BG, fg=MID_FG, anchor="w")
score_r2_lbl.pack(fill=tk.X, padx=10, pady=(0, 10))

driver_cell  = make_text_cell(frame2, height=5, row=1, col=2)
context_cell = make_text_cell(frame2, height=5, row=1, col=3)


def _draw_bar(colour):
    score_bar_cv.update_idletasks()
    w = score_bar_cv.winfo_width()
    if w < 4:
        w = 120
    s = _draw_bar._last_score
    score_bar_cv.delete("all")
    filled = max(2, int(w * score_pct(s)))
    score_bar_cv.create_rectangle(0, 0, w, 5, fill="#333", outline="")
    score_bar_cv.create_rectangle(0, 0, filled, 5, fill=colour, outline="")

_draw_bar._last_score = 0.0


def on_line_selected(event=None):
    line = line_cb.get()
    if line not in PRODUCTION_LINES: return
    s  = problem_score(PRODUCTION_LINES[line])
    cl = score_col(s)
    score_num_lbl.config(text=f"{s:.4f}" if s != float("inf") else "∞", fg=cl)
    score_rank_lbl.config(text=score_rank(s))
    _draw_bar._last_score = s
    root.after(30, lambda: _draw_bar(cl))
    ols_m = get_machine_ols(line)
    n     = len(df_flat[df_flat["machine_id"] == line])
    score_r2_lbl.config(
        text=(f"Machine OLS R² = {ols_m.rsquared:.3f}  (n = {n})"
              if ols_m else f"OLS unavailable  (n = {n})"))
    top_col, display, annot = machine_top_driver(line)
    fill_text(driver_cell, f"{display}\n\n{annot}")
    pathway = _CAUSAL_CONTEXT.get(top_col, "No pathway mapped in ontology")
    meds = [c for c in COL_META.get(top_col, {}).get("likely_downstream_effects", [])
            if c in SCRAP_PARENTS]
    if meds:
        pathway += f"\n\nMediated via: {', '.join(meds)}"
    fill_text(context_cell, pathway)


line_cb.bind("<<ComboboxSelected>>", on_line_selected)
_worst = max(PRODUCTION_LINES, key=lambda k: problem_score(PRODUCTION_LINES[k]))
line_cb.set(_worst)
on_line_selected()


# ══════════════════════════════════════════════════════
#  STATUS BAR
# ══════════════════════════════════════════════════════

status_var = tk.StringVar(value="⏳  LiNGAM causal discovery running in background…")
tk.Label(root, textvariable=status_var, font=FONT_SM,
         bg="#111", fg=MID_FG, anchor="w").pack(
    side=tk.BOTTOM, fill=tk.X, padx=0)

def _poll_lingam():
    if LINGAM_READY.is_set():
        if "error" in LINGAM_RESULTS:
            status_var.set(f"⚠  LiNGAM error: {LINGAM_RESULTS['error']}")
        else:
            order = LINGAM_RESULTS.get("order", [])
            status_var.set(
                "✓  LiNGAM ready  —  causal order: " +
                " → ".join(COL_TO_LABEL.get(c, c) for c in order))
        if var_cb.get():
            on_var_selected()
    else:
        root.after(500, _poll_lingam)

root.after(500, _poll_lingam)

root.mainloop()