import streamlit as st
import numpy as np
from scipy.stats import beta, chisquare
from scipy import integrate
import matplotlib.pyplot as plt

# --- Analytical Function for P(B > A) ---
def prob_B_superior(alpha_A, beta_A, alpha_B, beta_B):
    """
    Calculates the exact probability P(B > A) using numerical integration.
    """
    integrand = lambda y: beta.pdf(y, alpha_B, beta_B) * beta.cdf(y, alpha_A, beta_A)
    # The result is the integral of the integrand from 0 to 1
    result, _ = integrate.quad(integrand, 0, 1)
    return result

st.title("Bayesian Uplift Certainty Estimator")

st.markdown("""
This tool estimates the **certainty of uplift** between two variants (A and B) using Bayesian posterior inference.

Provide your test results below to get:
- A precise, analytical probability that **B is better than A**.
- The estimated **uplift and its credible interval**.
- Visualizations of the posterior distributions.
""")

# --- Interpretation Mode ---
mode = st.radio("Interpretation Strictness", ["Strict", "Lenient"], horizontal=True, help="Strict mode requires both high probability and the credible interval to be fully above 0 to conclude B is better. Lenient mode considers only the probability.")

# --- Credible Interval Setting ---
credibility = st.slider("Credible Interval (%)", min_value=90, max_value=99, value=95, step=1, help="Choose how confident you want to be in the credible interval. 95% is common.")

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

# --- Sample Ratio Mismatch Check using Chi-square Test ---
observed = [n_A, n_B]
total = n_A + n_B
expected = [total / 2, total / 2]
chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)

if p_value < 0.01:
    st.error("🚫 Sample Ratio Mismatch detected via Chi-square test (p < 0.01).")
    with st.expander("ℹ️ What is a Sample Ratio Mismatch?"):
        st.markdown("""
        A **Sample Ratio Mismatch (SRM)** occurs when the number of users allocated to each variant is significantly different from what you expected—typically a 50/50 split in an A/B test.

        This can indicate a problem with randomization, test assignment logic, or user targeting. SRM can bias your test results and reduce the validity of your conclusions.

        **Best Practice:** Pause the test, investigate the allocation logic, and re-run it once fixed. This test's results should be discarded as SRM breaks the assumptions of random assignment.
        """)


# --- Priors & Samples ---
alpha_prior = 1
beta_prior = 1
samples = 10000

# --- Posterior Calculations ---
# Define posterior alpha and beta values
alpha_A_post = alpha_prior + conv_A
beta_A_post = beta_prior + n_A - conv_A
alpha_B_post = alpha_prior + conv_B
beta_B_post = beta_prior + n_B - conv_B

# --- Main Calculations ---
# 1. Use the analytical function for a precise P(B > A)
prob_B_better = prob_B_superior(alpha_A_post, beta_A_post, alpha_B_post, beta_B_post)

# 2. Use simulation to get distributions for uplift and plotting
post_A = beta(alpha_A_post, beta_A_post)
post_B = beta(alpha_B_post, beta_B_post)
rng = np.random.default_rng(seed=42)
samples_A = post_A.rvs(samples, random_state=rng)
samples_B = post_B.rvs(samples, random_state=rng)

# Calculate uplift distribution and credible intervals from samples
uplift_samples = (samples_B - samples_A) / samples_A
mean_uplift = np.mean(uplift_samples)
ci_lower, ci_upper = np.percentile(
    uplift_samples,
    [(100 - credibility) / 2, 100 - (100 - credibility) / 2]
)

# --- Output ---
st.subheader("Results")

st.metric(label="Probability B is better than A", value=f"{prob_B_better:.2%}")
st.caption("Calculated analytically for high precision.")


if (mode == "Strict" and prob_B_better > 0.95 and ci_lower > 0) or (mode == "Lenient" and prob_B_better > 0.95):
    st.success("✅ This result is conclusive.")
elif prob_B_better > 0.90:
    st.info("ℹ️ There is moderate confidence that B is better.")
elif prob_B_better > 0.75:
    st.warning("⚠️ The evidence is weak or inconclusive.")
else:
    st.error("❌ The result suggests B may not be better than A.")

st.write(f"**Estimated Mean Uplift:** {mean_uplift:.2%}")
st.write(f"**{credibility}% Credible Interval for Uplift:** [{ci_lower:.2%}, {ci_upper:.2%}]")
st.caption("The uplift distribution is calculated via simulation.")


# --- Stakeholder Interpretation ---
st.subheader("Plain-Language Summary")
if ci_lower > 0:
    st.success(f"With a mean uplift of {mean_uplift:.2%} and the {credibility}% credible interval entirely above zero ({ci_lower:.2%} to {ci_upper:.2%}), it's highly likely that Variant B is performing better than Variant A.")
elif ci_upper < 0:
    st.error(f"The test suggests a negative uplift of {mean_uplift:.2%}. The {credibility}% credible interval is entirely below zero ({ci_lower:.2%} to {ci_upper:.2%}), strongly indicating that Variant B is likely performing worse than A.")
else:
    st.warning(f"The estimated mean uplift is {mean_uplift:.2%}, but the {credibility}% credible interval spans from {ci_lower:.2%} to {ci_upper:.2%}, which includes zero. This means we cannot be certain that Variant B is truly better or worse than A.")

# --- Plotting ---
st.subheader("Visualizations")
with st.expander("ℹ️ What do these graphs show?"):
    st.markdown(f"""
- **Posterior Distributions** (left): These curves represent our updated beliefs about the true conversion rates for Variant A and Variant B. The more separation you see between them, the stronger the evidence of a real difference.

- **Estimated Uplift Distribution** (right): This histogram shows the distribution of possible uplift values (how much better B is than A) based on the simulation. The black line is the estimated mean uplift, and the red dashed lines show the {credibility}% credible interval.
    """)

# Create a figure with two subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style

# --- 1. Posterior Distributions Plot (Left) ---
x = np.linspace(0, 1, 1000)

# Plot smooth PDFs for A and B
ax[0].plot(x, post_A.pdf(x), label='Variant A', color='royalblue')
ax[0].fill_between(x, post_A.pdf(x), alpha=0.3, color='royalblue')
ax[0].plot(x, post_B.pdf(x), label='Variant B', color='darkorange')
ax[0].fill_between(x, post_B.pdf(x), alpha=0.3, color='darkorange')

# Dynamically set the x-axis limits to zoom in on the interesting area
lower_bound = min(post_A.ppf(0.001), post_B.ppf(0.001))
upper_bound = max(post_A.ppf(0.999), post_B.ppf(0.999))
ax[0].set_xlim(lower_bound, upper_bound)

ax[0].set_title("Posterior Distributions", fontsize=14)
ax[0].set_xlabel("Conversion Rate", fontsize=12)
ax[0].set_ylabel("Density", fontsize=12)
ax[0].legend()
ax[0].xaxis.set_major_formatter(plt.FuncFormatter('{:.2%}'.format))


# --- 2. Estimated Uplift Distribution Plot (Right) ---
# Normalize histogram to show density
ax[1].hist(uplift_samples, bins=50, color='purple', alpha=0.7, density=True)

# Add vertical lines for credible interval and mean
ax[1].axvline(ci_lower, color='red', linestyle='--', label=f'{credibility}% Credible Interval')
ax[1].axvline(ci_upper, color='red', linestyle='--')
ax[1].axvline(mean_uplift, color='black', linestyle='-', label='Mean Uplift')

ax[1].set_title("Estimated Uplift Distribution", fontsize=14)
ax[1].set_xlabel("Relative Uplift", fontsize=12)
ax[1].legend()
ax[1].xaxis.set_major_formatter(plt.FuncFormatter('{:.1%}'.format))

# Improve layout and display the plot
fig.tight_layout(pad=3.0)
st.pyplot(fig)
