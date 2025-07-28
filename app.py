import streamlit as st
import numpy as np
from scipy.stats import beta, chisquare
import matplotlib.pyplot as plt

# 1. Set Page Configuration
st.set_page_config(
    page_title="Uplift Estimator | Bayesian Toolkit",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Core Calculation Function ---
@st.cache_data
def run_bayesian_analysis(n_A, conv_A, n_B, conv_B, credibility):
    """
    Performs the Bayesian analysis using simulation.
    Returns a dictionary of results.
    """
    alpha_prior, beta_prior = 1, 1
    samples = 20000
    
    alpha_A_post = alpha_prior + conv_A
    beta_A_post = beta_prior + n_A - conv_A
    alpha_B_post = alpha_prior + conv_B
    beta_B_post = beta_prior + n_B - conv_B

    post_A = beta(alpha_A_post, beta_A_post)
    post_B = beta(alpha_B_post, beta_B_post)
    
    rng = np.random.default_rng(seed=42)
    samples_A = post_A.rvs(samples, random_state=rng)
    samples_B = post_B.rvs(samples, random_state=rng)

    prob_B_better = np.mean(samples_B > samples_A)
    uplift_samples = (samples_B - samples_A) / samples_A
    mean_uplift = np.mean(uplift_samples)
    ci_lower, ci_upper = np.percentile(
        uplift_samples,
        [(100 - credibility) / 2, 100 - (100 - credibility) / 2]
    )
    
    return {
        "prob_B_better": prob_B_better,
        "mean_uplift": mean_uplift,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "post_A": post_A,
        "post_B": post_B,
        "uplift_samples": uplift_samples
    }

# 2. Page Title and Introduction
st.title("📈 Uplift Certainty Estimator")
st.markdown(
    "This tool helps you interpret A/B test results using Bayesian inference to determine the certainty of an uplift."
)

# 3. Sidebar for All User Inputs
with st.sidebar:
    st.header("Parameters")

    st.subheader("Test Results")
    n_A = st.number_input("Sample Size - Variant A", min_value=1, value=10000, step=100)
    conv_A = st.number_input("Conversions - Variant A", min_value=0, value=500, step=10)

    n_B = st.number_input("Sample Size - Variant B", min_value=1, value=10000, step=100)
    conv_B = st.number_input("Conversions - Variant B", min_value=0, value=550, step=10)
    
    st.subheader("Settings")
    mode = st.radio(
        "Interpretation Strictness", ["Strict", "Lenient"], horizontal=True,
        help="Strict mode requires P(B>A) > 95% AND the credible interval to be above zero. Lenient mode only considers the probability."
    )
    credibility = st.slider(
        "Credible Interval (%)", min_value=80, max_value=99, value=95, step=1,
        help="The confidence level for the uplift's credible interval."
    )

    st.markdown("---")
    run_button = st.button("Run Analysis", type="primary", use_container_width=True)

# 4. Main Page for Displaying Outputs
st.markdown("---")

if run_button:
    if conv_A > n_A or conv_B > n_B:
        st.error("Conversions cannot exceed the sample size for a variant.")
    else:
        observed = [n_A, n_B]
        total = n_A + n_B
        expected = [total / 2, total / 2]
        chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)

        if p_value < 0.01:
            st.error("🚫 **Sample Ratio Mismatch (SRM) Detected** (p < 0.01). Results may be unreliable.")
        
        with st.spinner("Running Bayesian analysis..."):
            results = run_bayesian_analysis(n_A, conv_A, n_B, conv_B, credibility)
            
            prob_B_better = results["prob_B_better"]
            ci_lower = results["ci_lower"]

            st.subheader("Results")
            st.metric(label="Probability B is better than A", value=f"{prob_B_better:.2%}")
            
            if (mode == "Strict" and prob_B_better > 0.95 and ci_lower > 0) or \
               (mode == "Lenient" and prob_B_better > 0.95):
                st.success("✅ This result is conclusive.")
            elif prob_B_better > 0.90:
                st.info("ℹ️ There is moderate confidence that B is better.")
            else:
                st.warning("⚠️ The evidence is weak or inconclusive.")

            st.write(f"**Estimated Mean Uplift:** {results['mean_uplift']:.2%}")
            st.write(f"**{credibility}% Credible Interval for Uplift:** [{ci_lower:.2%}, {results['ci_upper']:.2%}]")

            st.subheader("Plain-Language Summary")
            if ci_lower > 0:
                st.success(f"With a mean uplift of {results['mean_uplift']:.2%}, it's highly likely that Variant B is performing better than A. The entire {credibility}% credible interval is above zero, supporting a real positive improvement.")
            elif results['ci_upper'] < 0:
                st.error(f"The test suggests a negative uplift of {results['mean_uplift']:.2%}. The credible interval is entirely below zero, strongly indicating that Variant B is likely performing worse than A.")
            else:
                st.warning(f"The estimated uplift is {results['mean_uplift']:.2%}, but the credible interval includes zero. This means we cannot be certain that Variant B is truly better or worse than A.")

            st.subheader("Visualizations")
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            plt.style.use('seaborn-v0_8-whitegrid')

            post_A, post_B = results["post_A"], results["post_B"]
            x = np.linspace(0, 1, 1000)
            ax[0].plot(x, post_A.pdf(x), label='Variant A', color='royalblue')
            ax[0].fill_between(x, post_A.pdf(x), alpha=0.3, color='royalblue')
            ax[0].plot(x, post_B.pdf(x), label='Variant B', color='darkorange')
            ax[0].fill_between(x, post_B.pdf(x), alpha=0.3, color='darkorange')
            lower_bound = min(post_A.ppf(0.001), post_B.ppf(0.001))
            upper_bound = max(post_A.ppf(0.999), post_B.ppf(0.999))
            ax[0].set_xlim(lower_bound, upper_bound)
            ax[0].set_title("Posterior Distributions")
            ax[0].set_xlabel("Conversion Rate")
            ax[0].set_ylabel("Density")
            ax[0].legend()
            ax[0].xaxis.set_major_formatter(plt.FuncFormatter('{:.2%}'.format))

            ax[1].hist(results["uplift_samples"], bins=50, color='purple', alpha=0.7, density=True)
            ax[1].axvline(ci_lower, color='red', linestyle='--', label=f'{credibility}% Credible Interval')
            ax[1].axvline(results["ci_upper"], color='red', linestyle='--')
            ax[1].axvline(results["mean_uplift"], color='black', linestyle='-', label='Mean Uplift')
            ax[1].set_title("Estimated Uplift Distribution")
            ax[1].set_xlabel("Relative Uplift")
            ax[1].legend()
            ax[1].xaxis.set_major_formatter(plt.FuncFormatter('{:.1%}'.format))
            
            fig.tight_layout(pad=3.0)
            st.pyplot(fig)
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Analysis'.")

# 5. Explanations Section
st.markdown("---")
with st.expander("ℹ️ How to interpret these results"):
    st.markdown("""
    #### The Key Metrics
    * **Probability B is better than A:** This is the core Bayesian output. A value of 95% means there's a 95% chance that Variant B's true conversion rate is higher than Variant A's.
    * **Estimated Mean Uplift:** The average expected improvement of B over A based on the simulation.
    * **Credible Interval:** The range where we are confident the *true* uplift lies. If a 95% credible interval is `[1%, 5%]`, we're 95% certain the real uplift is in that positive range.

    ---
    #### The Visualizations
    * **Posterior Distributions (Left Graph):** These curves show our belief about the true conversion rate for each variant after seeing the data. **Look for separation:** the less the two curves overlap, the stronger the evidence for a real difference.
    * **Estimated Uplift Distribution (Right Graph):** This shows the range of possible uplift values. **Check if it crosses zero:** If the entire credible interval (the area between the red dashed lines) is above zero, it provides strong evidence that B is a true winner.

    ---
    #### The Plain-Language Summary
    This final section translates all the statistics above into a clear, actionable business recommendation, helping you decide whether to launch the change.
    """)
