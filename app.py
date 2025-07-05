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

# --- Interpretation Mode ---
mode = st.radio("Interpretation Strictness", ["Strict", "Lenient"], horizontal=True, help="Strict mode requires both high probability and credible interval fully above 0 to conclude B is better. Lenient mode considers only probability.")

# --- Credible Interval Setting ---
credibility = st.slider("Credible Interval (%)", min_value=90, max_value=99, value=95, step=1, help="Choose how confident you want to be in the credible interval. 95% is common, but you can lower it to make the interval narrower.")

# --- Inputs ---
st.subheader("Enter Your Results")
n_A = st.number_input("Sample size - Variant A", min_value=1, value=1000, step=1, key="n_A")
conv_A = st.number_input("Conversions - Variant A", min_value=0, value=min(50, int(n_A)), step=1, key="conv_A")

n_B = st.number_input("Sample size - Variant B", min_value=1, value=1000, step=1, key="n_B")
conv_B = st.number_input("Conversions - Variant B", min_value=0, value=min(60, int(n_B)), step=1, key="conv_B")

# --- Validation ---
if conv_A > n_A:
    st.error("Conversions for Variant A cannot exceed its sample size.")
    st.stop()
if conv_B > n_B:
    st.error("Conversions for Variant B cannot exceed its sample size.")
    st.stop()

# --- Sample Ratio Mismatch Check ---
expected_ratio = 0.5
actual_ratio = n_A / (n_A + n_B)
SRM_detected = abs(actual_ratio - expected_ratio) > 0.05
if SRM_detected:
    st.error("🚫 Sample Ratio Mismatch detected. This test should be considered inconclusive and must be re-run.")
    with st.expander("ℹ️ What is a Sample Ratio Mismatch?"):
        st.markdown("""
        A **Sample Ratio Mismatch (SRM)** occurs when the number of users allocated to each variant is significantly different from what you expected — typically a 50/50 split in an A/B test.

        This can indicate a problem with randomization, test assignment logic, or user targeting. SRM can bias your test results and reduce the validity of your conclusions.

        **Best Practice:** Pause the test, investigate the allocation logic, and re-run it once fixed. This test's results should be discarded. SRM breaks the assumptions of random assignment and renders the outcome invalid.
        """)


alpha_prior = 1
beta_prior = 1
samples = 10000


# --- Posteriors ---
post_A = beta(alpha_prior + conv_A, beta_prior + n_A - conv_A)
post_B = beta(alpha_prior + conv_B, beta_prior + n_B - conv_B)

rng = np.random.default_rng(seed=42)
samples_A = post_A.rvs(samples, random_state=rng)
samples_B = post_B.rvs(samples, random_state=rng)

# --- Calculations ---
uplift_samples = (samples_B - samples_A) / samples_A
prob_B_better = np.mean(samples_B > samples_A)
mean_uplift = np.mean(uplift_samples)
ci_lower, ci_upper = np.percentile(
    uplift_samples,
    [(100 - credibility) / 2, 100 - (100 - credibility) / 2]
)

# --- Output ---
st.subheader("Results")

st.write(f"**Probability B is better than A:** {prob_B_better:.2%}")

if (mode == "Strict" and prob_B_better > 0.95 and ci_lower > 0) or (mode == "Lenient" and prob_B_better > 0.95):
    st.success("✅ This result is conclusive.")
elif prob_B_better > 0.90:
    st.info("ℹ️ There is moderate confidence that B is better.")
elif prob_B_better > 0.75:
    st.warning("⚠️ The evidence is weak or inconclusive.")
else:
    st.error("❌ The result suggests B may not be better than A.")

st.write(f"**Estimated Mean Uplift:** {mean_uplift:.2%} 🛈")
st.caption("This is the average uplift between B and A based on thousands of sampled outcomes. A positive value indicates that B tends to perform better.")
st.write(f"**{credibility}% Credible Interval:** [{ci_lower:.2%}, {ci_upper:.2%}] 🛈")
st.caption("This is the range within which we believe the true uplift likely falls, based on the data and model. If the interval includes 0, there is uncertainty about whether B is truly better.")

# --- Stakeholder Interpretation ---
if ci_lower > 0:
    st.success(f"With a mean uplift of {mean_uplift:.2%} and a {credibility}% credible interval from {ci_lower:.2%} to {ci_upper:.2%}, it's highly likely that Variant B is performing better than Variant A. The entire interval is above 0, supporting a real and positive improvement.")
elif ci_upper < 0:
    st.error(f"The test suggests a mean uplift of {mean_uplift:.2%}, but the {credibility}% credible interval ranges from {ci_lower:.2%} to {ci_upper:.2%}, entirely below zero. This strongly indicates that Variant B is likely worse than A.")
else:
    st.warning(f"The estimated mean uplift is {mean_uplift:.2%}, but the {credibility}% credible interval spans from {ci_lower:.2%} to {ci_upper:.2%}, which includes zero. This means there's still uncertainty about whether Variant B is truly better or worse than A. More data may be needed to draw a clear conclusion.")

# --- Plotting ---

with st.expander("ℹ️ What do these graphs show?"):
    st.markdown(f"""
- **Posterior Distributions** (left): These histograms represent our updated beliefs about the true conversion rates for Variant A and Variant B, based on the observed data and priors. The more separation you see between the two curves, the stronger the evidence of a real difference.

- **Estimated Uplift Distribution** (right): This shows the distribution of possible uplift values (how much better B is than A). The black line is the estimated mean uplift, and the red dashed lines represent the {credibility}% credible interval.

These visualizations help you assess not just whether B is likely better, but also how *much* better it could be — and with what certainty.
    """)

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
