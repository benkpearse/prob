import streamlit as st
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

st.title("Bayesian Uplift Certainty Estimator")

st.markdown("""
This tool estimates the **certainty of uplift** between two variants (A and B) using Bayesian posterior inference.

Provide your test results below to get:
- Probability that **B is better than A**
- Estimated **uplift and credible interval**
- Visualization of posterior distributions
""")

# --- Inputs ---
st.subheader("Enter Your Results")
n_A = st.number_input("Sample size - Variant A", min_value=1, value=1000, step=1)
conv_A = st.number_input("Conversions - Variant A", min_value=0, max_value=n_A, value=50, step=1)

n_B = st.number_input("Sample size - Variant B", min_value=1, value=1000, step=1)
conv_B = st.number_input("Conversions - Variant B", min_value=0, max_value=n_B, value=60, step=1)

alpha_prior = 1
beta_prior = 1
samples = 10000

# --- Posteriors ---
post_A = beta(alpha_prior + conv_A, beta_prior + n_A - conv_A)
post_B = beta(alpha_prior + conv_B, beta_prior + n_B - conv_B)

samples_A = post_A.rvs(samples)
samples_B = post_B.rvs(samples)

# --- Calculations ---
uplift_samples = (samples_B - samples_A) / samples_A
prob_B_better = np.mean(samples_B > samples_A)
mean_uplift = np.mean(uplift_samples)
ci_lower, ci_upper = np.percentile(uplift_samples, [2.5, 97.5])  # 95% credible interval

# --- Output ---
st.subheader("Results")

st.write(f"**Probability B is better than A:** {prob_B_better:.2%}")

if prob_B_better > 0.95:
    st.success("✅ This result is conclusive.")
elif prob_B_better > 0.90:
    st.info("ℹ️ There is moderate confidence that B is better.")
elif prob_B_better > 0.75:
    st.warning("⚠️ The evidence is weak or inconclusive.")
else:
    st.error("❌ The result suggests B may not be better than A.")

st.write(f"**Estimated Mean Uplift:** {mean_uplift:.2%}")
st.caption("This is the average uplift between B and A based on thousands of sampled outcomes. A positive value indicates that B tends to perform better.")
st.write(f"**95% Credible Interval:** [{ci_lower:.2%}, {ci_upper:.2%}]")
st.caption("This is the range within which we believe the true uplift likely falls, based on the data and model. If the interval includes 0, there is uncertainty about whether B is truly better.")

# --- Stakeholder Interpretation ---
if ci_lower > 0:
    st.success(f"With a mean uplift of {mean_uplift:.2%} and a 95% credible interval from {ci_lower:.2%} to {ci_upper:.2%}, it's highly likely that Variant B is performing better than Variant A. The entire interval is above 0, supporting a real and positive improvement.")
elif ci_upper < 0:
    st.error(f"The test suggests a mean uplift of {mean_uplift:.2%}, but the 95% credible interval ranges from {ci_lower:.2%} to {ci_upper:.2%}, entirely below zero. This strongly indicates that Variant B is likely worse than A.")
else:
    st.warning(f"The estimated mean uplift is {mean_uplift:.2%}, but the 95% credible interval spans from {ci_lower:.2%} to {ci_upper:.2%}, which includes zero. This means there's still uncertainty about whether Variant B is truly better or worse than A. More data may be needed to draw a clear conclusion.")

# --- Plotting ---
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(samples_A, bins=50, alpha=0.6, label='A')
ax[0].hist(samples_B, bins=50, alpha=0.6, label='B')
ax[0].set_title("Posterior Distributions")
ax[0].legend()

ax[1].hist(uplift_samples, bins=50, color='purple')
ax[1].axvline(ci_lower, color='red', linestyle='--')
ax[1].axvline(ci_upper, color='red', linestyle='--')
ax[1].axvline(mean_uplift, color='black')
ax[1].set_title("Estimated Uplift Distribution")

st.pyplot(fig)
