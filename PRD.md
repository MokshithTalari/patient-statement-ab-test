# PRD: Patient Statement Redesign A/B Test (Simulation)

**Owner:** Mokshith Talari, Senior Business Analyst
**Status:** Complete (simulation)
**Data:** 100% synthetic. Portfolio case study; no client, employer, payer, or patient data.

## 1. Problem

In a hypothetical hospital system, self-pay balances settle slowly. Baseline (control parameters in `generate_dataset.py`):

| Baseline metric | Value |
|---|---|
| Mean days from statement to settlement | ~49 days |
| Paid in full within 30 days | ~9% of statements |
| Unpaid at 120 days | ~11.5% of statements |
| QR/mobile share of payments | ~22% |

Working hypothesis: statement confusion and payment friction, not unwillingness to pay, drive the delay.

## 2. Proposed change

A redesigned statement with (1) a plain-language breakdown of insurance coverage vs. patient balance, (2) a QR code linking to mobile payment, (3) a 3-click interest-free payment plan.

## 3. Success metrics (set before analysis)

| Type | Metric | Decision rule |
|---|---|---|
| Primary | Mean days to pay (paid statements) | Reduction >= 6 days, p < 0.05 two-sided |
| Secondary | Paid within 30 days | Increase >= 5 pp |
| Secondary | Unpaid at 120 days | Decrease, p < 0.05 |
| Secondary | QR/mobile share of paid statements | >= 45% |
| Guardrail | Dispute rate | No increase greater than 0.5 pp |
| Guardrail (production) | Payment gateway failure rate | < 0.5% |

## 4. Experiment design

| Element | Specification |
|---|---|
| Unit of randomization | Patient; all of a patient's statements receive the same template |
| Allocation | Bernoulli 50/50, seeded |
| Sample | 50,000 statements (~26,600 patients), 90-day issue window, 120-day observation window |
| Power | alpha 0.05 two-sided, power 0.80, MDE 1.2 days, SD ~14.5 -> ~2,300 statements/arm required |
| Validity checks | Sample-ratio mismatch (chi-square); pre-treatment balance (Welch t); age mix (chi-square) |
| Primary test | Welch's t-test; robustness: patient-level means (clustering) and Mann-Whitney U (skew) |
| Rate tests | Two-proportion z-test |

## 5. Data dictionary

| Column | Type | Description |
|---|---|---|
| `statement_id` | string | Unique statement ID, e.g. `STMT-100234` |
| `patient_id` | string | Synthetic patient ID, e.g. `PAT-28421` |
| `cohort_group` | string | `Control` or `Treatment` |
| `statement_date` | date | Statement issue date |
| `payment_date` | date | Settlement date; null if unpaid at 120 days |
| `days_to_pay` | float | Days from statement to settlement; null if unpaid |
| `patient_balance` | float | Patient out-of-pocket balance ($25-$2,500) |
| `balance_tier` | string | `Low (<$150)`, `Medium ($150-$500)`, `High (>$500)` |
| `primary_payor` | string | BlueCross BlueShield, UnitedHealthcare, Aetna, Medicare Advantage, Cigna |
| `patient_age_group` | string | `18-35`, `36-55`, `56+` |
| `paid_within_30_days` | int | 1 if settled within 30 days |
| `payment_channel` | string | `QR_Portal`, `Phone_IVR`, `Mail_Check`, `Unpaid` |
| `dispute_flag` | int | 1 if the patient disputed the statement |

## 6. Business case method

Scenario assumptions: 120,000 statements/year, mean balance from data, 8% cost of capital.
One-time cash acceleration = daily self-pay base x days saved. Annual carrying-cost saving = that amount x cost of capital.
Annual balances no longer unpaid = self-pay base x reduction in unpaid rate (gross, before collection costs).
One-time and annual components are reported separately and not summed. Results: see `README.md`.

## 7. Out of scope

Payer adjudication, contract terms, and insurer-side A/R. This test measures patient behavior after adjudication only.
