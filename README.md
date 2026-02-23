![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Bias Clinic and Sampling Variability

## Overview

Every dataset you'll encounter as a data scientist was collected by someone, using some method, for some purpose — and each of those choices can introduce bias. Selection bias, survivorship bias, measurement bias, and data leakage are not just textbook concepts; they routinely distort real analyses and lead to wrong conclusions. The first skill of a responsible data scientist is recognizing when data *can't* be taken at face value.

In this lab you'll work through two complementary tracks. First, you'll diagnose bias in realistic case studies — reading scenarios and identifying what went wrong in the data collection process. Second, you'll move to code: simulating biased and unbiased sampling from a known population so you can *see* how bias shifts estimates and how sampling variability behaves. You'll finish with a preview of bootstrapping — a resampling technique that lets you quantify uncertainty without relying on distributional assumptions — and a hands-on demonstration of data leakage in a prediction pipeline.

This lab connects to the lesson on sampling, bias, and data quality. Where the lesson introduced types of bias and the concept of a sampling distribution, here you'll build intuition through simulation and critical analysis.

## Learning Goals

By the end of this lab, you should be able to:

- Identify selection bias, survivorship bias, and measurement bias in real-world data scenarios.
- Simulate random, stratified, and biased sampling from a population with known parameters.
- Compare sample statistics to true population parameters across many repeated samples.
- Construct a basic bootstrap distribution and compute a bootstrap confidence interval.
- Demonstrate how data leakage inflates model performance and explain why it's dangerous.
- Articulate best practices for collecting representative, unbiased data.

## Setup and Context

You'll work inside a Jupyter Notebook for this lab. Part of the lab involves written analysis (the bias case studies), and part involves simulation code. Both belong in the same notebook, using markdown cells for analysis and code cells for simulation.

A synthetic population dataset is generated at the start so that you always know the "ground truth" — this lets you measure exactly how far sample estimates deviate from reality.

## Requirements

### Fork and clone

1. Fork this repository to your own GitHub account.
2. Clone the fork to your local machine.
3. Navigate into the project directory.

### Python environment

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

- **Python** ≥ 3.9
- **pandas** — data manipulation
- **numpy** — numerical operations and random sampling
- **matplotlib** — plotting
- **seaborn** — statistical visualization
- **scikit-learn** — used in the data leakage demonstration (Task 5)

## Getting Started

1. Create a new Jupyter Notebook called **`m3-03-bias-clinic-sampling-variability.ipynb`**.
2. Import cell:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)
sns.set_style("whitegrid")
```

3. Work through the tasks in order. Tasks 1–2 are analytical; Tasks 3–5 are code-heavy.
4. Use markdown cells liberally to explain your reasoning and observations.

## Tasks

### Task 1: Bias Diagnosis — Case Studies

For each of the following scenarios, identify the **type of bias** present (selection bias, survivorship bias, measurement bias, response bias, or another relevant type), explain **how** it distorts the conclusions, and suggest **one concrete fix**.

**Scenario A — App Review Analysis:**
A product team analyzes their app's reviews on the App Store to understand user satisfaction. The average rating is 4.3/5, so they conclude that most users are happy. They plan to reduce investment in customer support.

**Scenario B — Startup Success Study:**
A business school studies 200 successful tech startups (all founded in the last 10 years and still operating) to identify common traits that predict startup success. They find that 80% had a pivot in their first two years and conclude that pivoting is a key success strategy.

**Scenario C — Health Survey:**
A health organization sends a voluntary online survey to 50,000 email subscribers asking about exercise habits and health outcomes. The survey receives 5,000 responses (10% response rate). Results show that respondents exercise an average of 5 hours per week and have excellent self-reported health.

**Scenario D — Salary Benchmarking:**
A recruiting platform publishes average salaries by job title based on user-submitted salary data. The platform is popular among tech workers in large cities. A company in a small town uses this data to set their salary bands.

Write your analysis in markdown cells — no code required for this task. Aim for 3–5 sentences per scenario.

### Task 2: Create the Population

Before you can study sampling, you need a known population to sample from.

1. Generate a synthetic population of **100,000** individuals with the following columns:
   - `age`: integers drawn from a realistic distribution (e.g., a clipped normal centered at 40 with SD of 15, range 18–85).
   - `income`: correlated with age — use a linear relationship with noise (e.g., `income = 1500 * age + normal_noise`). Clip to a realistic range (e.g., 15,000–250,000).
   - `satisfaction`: a score from 1–10 that depends on income (higher income → slightly higher satisfaction, with plenty of noise).
   - `region`: categorical — randomly assign "Urban" (60%), "Suburban" (25%), "Rural" (15%).
2. Compute and store the **true population parameters**: mean age, mean income, mean satisfaction, and the proportion in each region.
3. Display the population summary and a grid of histograms (one per numerical variable).

These population parameters are your ground truth — everything in the following tasks is measured against them.

### Task 3: Biased vs. Unbiased Sampling

Draw samples from your population using three different strategies and compare how well each recovers the true population parameters.

1. **Simple random sample** (n = 200): every individual has an equal chance of being selected.
2. **Biased sample — Urban only** (n = 200): only sample from individuals in the "Urban" region.
3. **Biased sample — High-income filter** (n = 200): only sample from individuals with income above the population median.

For each sample:
- Compute mean age, mean income, and mean satisfaction.
- Display the results alongside the true population parameters in a comparison table.
- Create overlapping KDE plots (sample vs. population) for income and satisfaction.

Now, repeat each sampling strategy **1,000 times** and collect the sample means:
- Plot the **sampling distribution of the mean income** for each strategy (three histograms, side by side).
- Mark the true population mean on each histogram.

**Guiding question:** Which sampling strategies produce biased estimates? How can you tell from the sampling distributions?

### Task 4: Bootstrap Preview

Bootstrapping lets you estimate the variability of a statistic when you only have one sample and no knowledge of the underlying distribution.

1. Draw a single **simple random sample** of n = 100 from the population.
2. From this sample, generate **5,000 bootstrap resamples** (sample with replacement, same size as the original).
3. For each bootstrap resample, compute the **mean income**.
4. Plot the bootstrap distribution of mean income as a histogram.
5. Compute the **95% bootstrap confidence interval** using the percentile method (2.5th and 97.5th percentiles).
6. Check whether the true population mean income falls inside your bootstrap CI.

Repeat the entire process (steps 1–6) for **three different original sample sizes**: n = 30, n = 100, n = 500.
- Create a figure with three side-by-side histograms showing the bootstrap distributions.
- Mark the 95% CI boundaries and the true population mean on each.

**Guiding question:** How does the original sample size affect the width of the bootstrap confidence interval? Does the CI always contain the true mean?

### Task 5: Data Leakage Demonstration

Data leakage occurs when information from outside the training data "leaks" into the model, producing unrealistically good performance that won't generalize.

1. **Create a classification dataset:**
   - Use the population from Task 2. Create a binary target: `high_satisfaction = 1 if satisfaction >= 7 else 0`.
   - Features: `age`, `income`.

2. **The wrong way (leakage):**
   - Apply `StandardScaler` to the **entire dataset** (fit on all rows).
   - Split into train (80%) and test (20%).
   - Train a `LogisticRegression` on the training set.
   - Evaluate accuracy on the test set.

3. **The right way (no leakage):**
   - Split into train (80%) and test (20%) **first**.
   - Fit `StandardScaler` on the training set only; transform both train and test.
   - Train `LogisticRegression` on the training set.
   - Evaluate accuracy on the test set.

4. **Compare and discuss:**
   - Report both accuracy scores side by side.
   - In a markdown cell, explain why the "wrong way" leaks information and why the difference in accuracy might be small here but catastrophic in other scenarios (e.g., time series, feature engineering from target).

5. **Bonus — A more dramatic leakage example:**
   - Add a "future" feature to the dataset: `future_satisfaction_change = satisfaction + noise`. This feature is computed using the target variable.
   - Retrain both pipelines (leaky and correct) with this new feature included.
   - Compare accuracies — the leaky pipeline should show a dramatic performance gap.

**Guiding question:** Why is data leakage often harder to detect than other bugs? What practices can prevent it?

### Task 6: Reflection — Data Quality Principles

In a final markdown cell, write **5 principles** for responsible data collection and analysis that you would follow in a professional data science role. Each principle should be a short paragraph (2–3 sentences) that connects to something you observed in this lab. For example, one principle might address always checking for representativeness before trusting sample statistics.

## Submission

### What to submit

- `m3-03-bias-clinic-sampling-variability.ipynb` — your completed notebook with all code, outputs, and markdown explanations.

### Definition of done (checklist)

- [ ] All four bias case studies are analyzed with type, explanation, and fix.
- [ ] Population is generated with 100,000 rows and true parameters are stored.
- [ ] Three sampling strategies are compared across 1,000 repeated samples.
- [ ] Bootstrap confidence intervals are computed for three sample sizes.
- [ ] Data leakage demonstration shows both the wrong and right approaches.
- [ ] Reflection contains 5 concrete, lab-connected principles.
- [ ] The notebook runs top-to-bottom without errors (`Kernel → Restart & Run All`).

### How to submit (Git workflow)

```bash
git add .
git commit -m "lab: complete bias clinic and sampling variability"
git push origin main
```

Then open a **Pull Request** on the original repository with a brief description of your work.
