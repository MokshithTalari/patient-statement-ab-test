"""
generate_executive_infographic.py
---------------------------------
One-slide executive summary of the SIMULATED A/B test. Every number is read
from results.json (produced by run_analysis.py) or computed from the data.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

ROOT = Path(__file__).resolve().parent
r = json.loads((ROOT / "results.json").read_text())
df = pd.read_csv(ROOT / "data" / "patient_billing_cohorts.csv")
paid = df.dropna(subset=["days_to_pay"])
ARMS = ["Control", "Treatment"]
BG, CARD, EDGE, TXT, MUTED = "#0f172a", "#1e293b", "#334155", "#f8fafc", "#94a3b8"


def style(ax, title):
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_color(EDGE)
    ax.grid(color=EDGE, ls="--", alpha=0.5)
    ax.tick_params(colors=MUTED)
    ax.set_title(title, color=TXT, fontsize=13, fontweight="bold", pad=10)


fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[0.9, 2.0, 1.8], hspace=0.42, wspace=0.28)

d, res, unp, bc = r["days"], r["resolution_30d"], r["unpaid"], r["business_case"]
kpis = [
    ("DAYS TO PAY (TREATMENT)", f"{d['mean']['Treatment']:.1f} d",
     f"{d['diff']:.1f} d faster (95% CI {d['ci'][0]:.1f}-{d['ci'][1]:.1f})", "#38bdf8"),
    ("PAID WITHIN 30 DAYS", f"{100*res['rate']['Treatment']:.1f}%",
     f"{res['diff_pp']:+.1f} pp vs control ({100*res['rate']['Control']:.1f}%)", "#4ade80"),
    ("QR / MOBILE SHARE", f"{r['channel_share']['Treatment']['QR_Portal']:.1f}%",
     f"vs {r['channel_share']['Control']['QR_Portal']:.1f}% control (paid stmts)", "#a855f7"),
    ("UNPAID AT 120 DAYS", f"{100*unp['rate']['Treatment']:.1f}%",
     f"{unp['diff_pp']:+.1f} pp vs control", "#facc15"),
]
for i, (lab, val, sub, col) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i]); ax.set_facecolor(CARD); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(EDGE)
    ax.text(0.5, 0.78, lab, color=MUTED, fontsize=11, fontweight="bold", ha="center")
    ax.text(0.5, 0.42, val, color=col, fontsize=24, fontweight="heavy", ha="center")
    ax.text(0.5, 0.14, sub, color="#e2e8f0", fontsize=9.5, ha="center")

ax1 = fig.add_subplot(gs[1, 0:2]); style(ax1, "Days to Pay Distribution")
for a, col in zip(ARMS, ["#ef4444", "#38bdf8"]):
    s = paid.loc[paid.cohort_group == a, "days_to_pay"]
    sns.kdeplot(s, ax=ax1, color=col, fill=True, alpha=0.4, lw=2.5, clip=(0, 120),
                label=f"{a} mean ({s.mean():.1f} d)")
    ax1.axvline(s.mean(), color=col, ls=":", lw=2)
ax1.set_xlabel("Days from statement to payment", color=MUTED); ax1.set_ylabel("Density", color=MUTED)
ax1.legend(facecolor=BG, edgecolor=EDGE, labelcolor=TXT)

ax2 = fig.add_subplot(gs[1, 2:4]); style(ax2, "Paid Within 30 Days, by Age Group")
ages = sorted(r["age_resolution_pct"]); x = np.arange(len(ages)); w = 0.35
for off, a, col in [(-w/2, "Control", "#64748b"), (w/2, "Treatment", "#4ade80")]:
    vals = [r["age_resolution_pct"][g][a] for g in ages]
    bars = ax2.bar(x + off, vals, w, color=col, edgecolor=EDGE, label=a)
    for bar in bars:
        ax2.text(bar.get_x() + w/2, bar.get_height() + 0.4, f"{bar.get_height():.1f}%",
                 ha="center", color=col, fontsize=10, fontweight="bold")
ax2.set_xticks(x); ax2.set_xticklabels(ages); ax2.set_ylabel("% of statements", color=MUTED)
ax2.set_ylim(0, max(max(v.values()) for v in r["age_resolution_pct"].values()) * 1.3)
ax2.legend(facecolor=BG, edgecolor=EDGE, labelcolor=TXT, loc="upper right")

ax3 = fig.add_subplot(gs[2, 0:2]); style(ax3, "Payment Channel Mix (Paid Statements)")
ch = ["QR_Portal", "Phone_IVR", "Mail_Check"]; xc = np.arange(3)
for off, a, col in [(-w/2, "Control", "#64748b"), (w/2, "Treatment", "#a855f7")]:
    ax3.bar(xc + off, [r["channel_share"][a][c] for c in ch], w, color=col, edgecolor=EDGE, label=a)
ax3.set_xticks(xc); ax3.set_xticklabels(["QR / Mobile", "Phone IVR", "Mail Check"])
ax3.set_ylabel("% of paid statements", color=MUTED); ax3.set_ylim(0, 65)
ax3.legend(facecolor=BG, edgecolor=EDGE, labelcolor=TXT, loc="upper center", ncol=2)

ax4 = fig.add_subplot(gs[2, 2:4]); style(ax4, "Illustrative Business Case (120K statements/yr, $K)")
cats = ["One-time cash\nacceleration", "Annual balances no\nlonger unpaid (gross)", "Annual carrying-cost\nsaving (8%)"]
vals = [bc["one_time_cash_acceleration"] / 1e3, bc["annual_balances_no_longer_unpaid"] / 1e3,
        bc["annual_carrying_cost_saving"] / 1e3]
bars = ax4.barh(cats, vals, color=["#38bdf8", "#4ade80", "#facc15"], edgecolor=EDGE, height=0.55)
for b in bars:
    ax4.text(b.get_width() + 15, b.get_y() + b.get_height() / 2, f"${b.get_width():,.0f}K",
             va="center", color=TXT, fontsize=10.5, fontweight="bold")
ax4.set_xlim(0, max(vals) * 1.3); ax4.tick_params(axis="y", colors=TXT)
ax4.set_xlabel("Scenario assumptions in run_analysis.py; not additive", color=MUTED)

fig.suptitle("PATIENT STATEMENT REDESIGN: A/B TEST SIMULATION\n"
             "Portfolio case study by Mokshith Talari  |  Synthetic data, illustrative results",
             color=TXT, fontsize=16, fontweight="heavy", y=0.99)
out = ROOT / "figures" / "executive_summary_infographic.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Wrote {out.relative_to(ROOT)}")
