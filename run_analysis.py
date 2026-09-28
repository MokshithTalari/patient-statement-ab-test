"""
run_analysis.py
---------------
End-to-end analysis of the SIMULATED patient-statement A/B test.
Writes every reported number to results.json so the dashboard, infographic,
README and notebook all read from one source (no hand-typed figures).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "patient_billing_cohorts.csv"
FIG = ROOT / "figures"

# ---- Illustrative business-case assumptions (edit to fit a real organization)
ASSUMPTIONS = {
    "annual_statement_volume": 120_000,
    "cost_of_capital": 0.08,          # annual rate applied to cash released
    "note": "Illustrative scenario only; not derived from any real organization.",
}

sns.set_theme(style="whitegrid", palette="deep")
ARMS = ["Control", "Treatment"]


def two_prop_z(x1, n1, x2, n2):
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (x1 / n1 - x2 / n2) / se
    return z, 2 * stats.norm.sf(abs(z))


def analyze(df: pd.DataFrame) -> dict:
    r = {"n_statements": {a: int((df.cohort_group == a).sum()) for a in ARMS}}
    pats = df.groupby("patient_id").cohort_group.first()
    r["n_patients"] = {a: int((pats == a).sum()) for a in ARMS}

    # 1. Sample-ratio mismatch (patient level, expected 50/50)
    chi_srm, p_srm = stats.chisquare([r["n_patients"][a] for a in ARMS])
    r["srm"] = {"chi2": chi_srm, "p": p_srm}

    # 2. Pre-treatment covariate balance
    b = [df.loc[df.cohort_group == a, "patient_balance"] for a in ARMS]
    t_b, p_b = stats.ttest_ind(*b, equal_var=False)
    r["balance_check"] = {"mean": {a: x.mean() for a, x in zip(ARMS, b)},
                          "sd": {a: x.std() for a, x in zip(ARMS, b)}, "t": t_b, "p": p_b}
    age_ct = pd.crosstab(df.cohort_group, df.patient_age_group)
    r["age_balance_p"] = stats.chi2_contingency(age_ct)[1]

    # 3. Primary metric: days to pay (paid statements)
    paid = df.dropna(subset=["days_to_pay"])
    c, t = (paid.loc[paid.cohort_group == a, "days_to_pay"] for a in ARMS)
    welch = stats.ttest_ind(c, t, equal_var=False)
    ci = welch.confidence_interval(0.95)
    r["days"] = {"mean": {"Control": c.mean(), "Treatment": t.mean()},
                 "median": {"Control": c.median(), "Treatment": t.median()},
                 "diff": c.mean() - t.mean(), "ci": [ci.low, ci.high],
                 "t": welch.statistic, "p": welch.pvalue}
    u = stats.mannwhitneyu(c, t, alternative="two-sided")
    r["days"]["mwu_p"] = u.pvalue
    # Cluster-robust check: collapse to one mean per patient
    pm = paid.groupby(["patient_id", "cohort_group"]).days_to_pay.mean().reset_index()
    pc, pt = (pm.loc[pm.cohort_group == a, "days_to_pay"] for a in ARMS)
    pw = stats.ttest_ind(pc, pt, equal_var=False)
    r["days"]["patient_level"] = {"diff": pc.mean() - pt.mean(), "t": pw.statistic, "p": pw.pvalue}

    # 4. Secondary metrics (all statements in denominator)
    def rate(col, val=None):
        out = {}
        for a in ARMS:
            s = df.loc[df.cohort_group == a, col]
            x = int((s == val).sum()) if val is not None else int(s.sum())
            out[a] = (x, len(s))
        z, p = two_prop_z(out["Treatment"][0], out["Treatment"][1], out["Control"][0], out["Control"][1])
        return {"rate": {a: out[a][0] / out[a][1] for a in ARMS},
                "diff_pp": 100 * (out["Treatment"][0] / out["Treatment"][1] - out["Control"][0] / out["Control"][1]),
                "z": z, "p": p}
    r["resolution_30d"] = rate("paid_within_30_days")
    r["unpaid"] = rate("payment_channel", "Unpaid")
    r["dispute"] = rate("dispute_flag")

    # Digital share among PAID statements
    r["channel_share"] = {a: (paid.loc[paid.cohort_group == a, "payment_channel"]
                              .value_counts(normalize=True).mul(100).round(1).to_dict()) for a in ARMS}

    # 5. Subgroups
    r["age_days_delta"] = (paid.groupby(["patient_age_group", "cohort_group"]).days_to_pay.mean()
                           .unstack().pipe(lambda x: (x.Control - x.Treatment).round(2)).to_dict())
    r["age_mean_days"] = (paid.groupby(["patient_age_group", "cohort_group"]).days_to_pay.mean()
                          .round(2).unstack().to_dict(orient="index"))
    r["age_resolution_pct"] = (df.groupby(["patient_age_group", "cohort_group"]).paid_within_30_days.mean()
                               .mul(100).round(1).unstack().to_dict(orient="index"))
    r["tier_days_delta"] = (paid.groupby(["balance_tier", "cohort_group"]).days_to_pay.mean()
                            .unstack().pipe(lambda x: (x.Control - x.Treatment).round(2)).to_dict())

    # 6. Illustrative business case (clearly separated one-time vs annual)
    vol = ASSUMPTIONS["annual_statement_volume"]
    base = vol * df.patient_balance.mean()
    one_time_cash = base / 365 * r["days"]["diff"]
    r["business_case"] = {
        **ASSUMPTIONS,
        "annual_self_pay_base": base,
        "one_time_cash_acceleration": one_time_cash,
        "annual_carrying_cost_saving": one_time_cash * ASSUMPTIONS["cost_of_capital"],
        "annual_balances_no_longer_unpaid": base * -r["unpaid"]["diff_pp"] / 100,
    }
    return r


def figures(df: pd.DataFrame, r: dict) -> None:
    FIG.mkdir(exist_ok=True)
    paid = df.dropna(subset=["days_to_pay"])

    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=paid, x="days_to_pay", hue="cohort_group", hue_order=ARMS,
                common_norm=False, fill=True, alpha=0.35, linewidth=2, clip=(0, 120))
    for a, col in zip(ARMS, ["#1f77b4", "#ff7f0e"]):
        plt.axvline(r["days"]["mean"][a], color=col, ls="--")
    plt.title("Days to Pay: Control vs Treatment (simulated data)", fontweight="bold")
    plt.xlabel("Days from statement to settlement"); plt.ylabel("Density")
    plt.legend([f"Treatment mean {r['days']['mean']['Treatment']:.1f} d",
                f"Control mean {r['days']['mean']['Control']:.1f} d"], title="Arm")
    plt.tight_layout(); plt.savefig(FIG / "figure1_distribution_days_to_pay.png", dpi=200); plt.close()

    plt.figure(figsize=(9, 5))
    agg = df.groupby(["patient_age_group", "cohort_group"]).paid_within_30_days.mean().mul(100).reset_index()
    ax = sns.barplot(data=agg, x="patient_age_group", y="paid_within_30_days", hue="cohort_group",
                     hue_order=ARMS, palette="Set2")
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height() + 0.5),
                        ha="center", fontsize=10, fontweight="bold")
    plt.title("Paid in Full Within 30 Days, by Age Group (simulated data)", fontweight="bold")
    plt.xlabel("Age group"); plt.ylabel("% of statements"); plt.ylim(0, agg.paid_within_30_days.max() * 1.25)
    plt.legend(title="Arm"); plt.tight_layout()
    plt.savefig(FIG / "figure2_resolution_rates_by_age.png", dpi=200); plt.close()

    mix = pd.DataFrame(r["channel_share"]).T[["QR_Portal", "Phone_IVR", "Mail_Check"]]
    ax = mix.plot(kind="bar", stacked=True, figsize=(8, 5), edgecolor="black", colormap="tab10")
    plt.title("Payment Channel Mix Among Paid Statements (simulated data)", fontweight="bold")
    plt.ylabel("% of paid statements"); plt.xlabel(""); plt.xticks(rotation=0)
    plt.legend(title="Channel", bbox_to_anchor=(1.02, 1), loc="upper left"); plt.tight_layout()
    plt.savefig(FIG / "figure3_channel_shift.png", dpi=200); plt.close()

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=paid, x="balance_tier", y="days_to_pay", hue="cohort_group", hue_order=ARMS,
                order=["Low (<$150)", "Medium ($150-$500)", "High (>$500)"], palette="Blues", showfliers=False)
    plt.title("Days to Pay by Balance Tier (simulated data, outliers hidden)", fontweight="bold")
    plt.xlabel("Balance tier"); plt.ylabel("Days to pay"); plt.legend(title="Arm"); plt.tight_layout()
    plt.savefig(FIG / "figure4_balance_tier_boxplot.png", dpi=200); plt.close()


def report(r: dict) -> None:
    d = r["days"]
    print(f"Patients: {r['n_patients']} | Statements: {r['n_statements']} | SRM p={r['srm']['p']:.3f}")
    print(f"Balance check p={r['balance_check']['p']:.3f} | Age balance p={r['age_balance_p']:.3f}")
    print(f"Days to pay: C {d['mean']['Control']:.2f} vs T {d['mean']['Treatment']:.2f} | "
          f"diff {d['diff']:.2f} [95% CI {d['ci'][0]:.2f}, {d['ci'][1]:.2f}] | Welch p={d['p']:.2e}")
    print(f"  Patient-level (cluster-robust) diff {d['patient_level']['diff']:.2f}, p={d['patient_level']['p']:.2e}")
    for k in ["resolution_30d", "unpaid", "dispute"]:
        m = r[k]
        print(f"{k}: C {100*m['rate']['Control']:.2f}% vs T {100*m['rate']['Treatment']:.2f}% "
              f"({m['diff_pp']:+.2f} pp, p={m['p']:.2e})")
    bc = r["business_case"]
    print(f"Business case (illustrative): one-time cash ${bc['one_time_cash_acceleration']:,.0f}; "
          f"annual carrying-cost ${bc['annual_carrying_cost_saving']:,.0f}; "
          f"annual balances no longer unpaid ${bc['annual_balances_no_longer_unpaid']:,.0f}")


if __name__ == "__main__":
    df = pd.read_csv(DATA)
    res = analyze(df)
    (ROOT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    figures(df, res)
    report(res)
