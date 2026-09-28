"""
generate_dataset.py
-------------------
Generates a SYNTHETIC patient-billing dataset for a simulated A/B test of a
redesigned patient statement. No real patient, payer, client, or employer data
is used. The treatment effects are parameters set below; the analysis in
run_analysis.py demonstrates how to design, test, and report such an
experiment, not a real-world finding.

Design choices:
  * Unit of randomization = PATIENT (not statement), so a patient with several
    statements always receives the same template (no cross-arm contamination).
  * Assignment is Bernoulli(0.5) per patient via a seeded numpy Generator, so
    the dataset is reproducible across numpy versions.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "patient_billing_cohorts.csv"

# ---- Simulation parameters (these ARE the "true" effects the test should recover)
CONTROL = dict(mean_days=48.5, sd_days=14.5, p_unpaid=0.115, p_dispute=0.034,
               channel_p={"Mail_Check": 0.46, "Phone_IVR": 0.32, "QR_Portal": 0.22})
TREATMENT = dict(mean_days=42.3, sd_days=13.8, p_unpaid=0.082, p_dispute=0.029,
                 channel_p={"Mail_Check": 0.21, "Phone_IVR": 0.26, "QR_Portal": 0.53})
AGE_BONUS_TREATMENT = {"18-35": 1.8, "36-55": 0.8, "56+": 0.0}  # extra days faster
OBSERVATION_WINDOW_DAYS = 120  # unpaid = not settled within this window


def generate_cohort_data(num_records: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Patients and patient-level randomization
    patient_ids = rng.integers(10_000, 45_000, size=num_records)
    unique_pats = np.unique(patient_ids)
    arm_by_patient = dict(zip(unique_pats,
                              np.where(rng.random(len(unique_pats)) < 0.5, "Control", "Treatment")))
    cohort = np.array([arm_by_patient[p] for p in patient_ids])

    statement_dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 90, num_records), unit="D")
    balances = np.round(np.clip(rng.lognormal(5.1, 0.75, num_records), 25.0, 2500.0), 2)
    tiers = np.select([balances < 150, balances <= 500], ["Low (<$150)", "Medium ($150-$500)"], "High (>$500)")
    payors = rng.choice(["BlueCross BlueShield", "UnitedHealthcare", "Aetna", "Medicare Advantage", "Cigna"],
                        size=num_records, p=[0.35, 0.25, 0.18, 0.14, 0.08])
    ages = rng.choice(["18-35", "36-55", "56+"], size=num_records, p=[0.28, 0.44, 0.28])

    is_t = cohort == "Treatment"
    p_unpaid = np.where(is_t, TREATMENT["p_unpaid"], CONTROL["p_unpaid"])
    p_disp = np.where(is_t, TREATMENT["p_dispute"], CONTROL["p_dispute"])
    unpaid = rng.random(num_records) < p_unpaid
    dispute = (rng.random(num_records) < p_disp).astype(int)

    age_bonus = np.where(is_t, pd.Series(ages).map(AGE_BONUS_TREATMENT).to_numpy(), 0.0)
    mu = np.where(is_t, TREATMENT["mean_days"], CONTROL["mean_days"]) - age_bonus
    sd = np.where(is_t, TREATMENT["sd_days"], CONTROL["sd_days"])
    slope = np.where(is_t, 0.004, 0.005)  # larger balances pay slightly slower
    days = np.clip(np.round(rng.normal(mu, sd) + balances * slope), 1, OBSERVATION_WINDOW_DAYS)

    def draw_channels(pmap, n):
        return rng.choice(list(pmap.keys()), size=n, p=list(pmap.values()))
    channel = np.empty(num_records, dtype=object)
    channel[~is_t] = draw_channels(CONTROL["channel_p"], (~is_t).sum())
    channel[is_t] = draw_channels(TREATMENT["channel_p"], is_t.sum())

    days = np.where(unpaid, np.nan, days)
    channel = np.where(unpaid, "Unpaid", channel)
    pay_dates = pd.Series(statement_dates) + pd.to_timedelta(days, unit="D")

    return pd.DataFrame({
        "statement_id": [f"STMT-{100000 + i}" for i in range(num_records)],
        "patient_id": [f"PAT-{p}" for p in patient_ids],
        "cohort_group": cohort,
        "statement_date": statement_dates.strftime("%Y-%m-%d"),
        "payment_date": pay_dates.dt.strftime("%Y-%m-%d"),
        "days_to_pay": days,
        "patient_balance": balances,
        "balance_tier": tiers,
        "primary_payor": payors,
        "patient_age_group": ages,
        "paid_within_30_days": ((~unpaid) & (days <= 30)).astype(int),
        "payment_channel": channel,
        "dispute_flag": dispute,
    })


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = generate_cohort_data()
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} synthetic rows -> {OUT.relative_to(ROOT)}")
    print(df["cohort_group"].value_counts().to_string())
