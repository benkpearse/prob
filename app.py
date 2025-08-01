import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import beta, chisquare
import matplotlib.pyplot as plt

# 1. Set Page Configuration
st.set_page_config(
    page_title="Uplift Estimator | Bayesian Toolkit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Core Calculation Functions ---
@st.cache_data
def run_multivariant_analysis(variant_data, credibility):
    """
    Performs Bayesian analysis for multiple variants.
    Returns a DataFrame of results and a list of posterior objects.
    """
    alpha_prior, beta_prior = 1, 1
    samples = 30000
    num_variants = len(variant_data)
    
    posteriors = []
    for data in variant_data:
        alpha_post = alpha_prior + data['conversions']
        beta_post = beta_prior + data['users'] - data['conversions']
        posteriors.append(beta(alpha_post, beta_post))

    rng = np.random.default_rng(seed=42)
    posterior_samples = [p.rvs(size=samples, random_state=rng) for p in posteriors]
    
    stacked_samples = np.stack(posterior_samples)
    best_variant_indices = np.argmax(stacked_samples, axis=0)
    prob_to_be_best = [np.mean(best_variant_indices == i) for i in range(num_variants)]

    control_samples = posterior_samples[0]
    results = []
    for i in range(num_variants):
        variant_samples = posterior_samples[i]
        
        uplift_samples = (variant_samples - control_samples) / control_samples
        mean_uplift = np.mean(uplift_samples)
        ci_lower, ci_upper = np.percentile(
            uplift_samples,
            [(100 - credibility) / 2, 100 - (100 - credibility) / 2]
        )

        results.append({
            "Variant": variant_data[i]['name'],
            "Users": variant_data[i]['users'],
            "Conversions": variant_data[i]['conversions'],
            "Conversion Rate": (variant_data[i]['conversions'] / variant_data[i]['users']) if variant_data[i]['users'] > 0 else 0,
            "Prob. to be Best": prob_to_be_best[i],
            "Uplift vs. Control": mean_uplift,
            "Credible Interval": (ci_lower, ci_upper)
        })

    results_df = pd.DataFrame(results)
    return results_df, posteriors

# --- Plotting Function ---
def plot_posteriors(posteriors, names):
    """
    Generates a plot of the posterior distributions for all variants.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    x = np.linspace(0, 1, 1000)
    colors = plt.cm.viridis(np.linspace(0, 1, len(posteriors)))
    
    all_ppf_low = [p.ppf(0.001) for p in posteriors]
    all_ppf_high = [p.ppf(0.999) for p in posteriors]

    for i, post in enumerate(posteriors):
        ax.plot(x, post.pdf(x), label=names[i], color=colors[i])
        ax.fill_between(x, post.pdf(x), alpha=0.3, color=colors[i])

    ax.set_xlim(min(all_ppf_low), max(all_ppf_high))
    ax.set_title("Posterior Distributions")
    ax.set_xlabel("Conversion Rate")
    ax.set_ylabel("Density")
    ax.legend()
    ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.2%}'.format))
    
    return fig

# --- Example Data Function ---
def load_example_data():
    st.session_state.num_variants = 3
    st.session_state.example_users = [10000, 10000, 10000]
    st.session_state.example_conversions = [500, 550, 520]

# 2. Page Title and Introduction
st.title("📈 Multi-Variant Uplift Estimator")
st.markdown("This tool interprets A/B/n test results using Bayesian inference to find the best performing variant.")

# 3. Sidebar for All User Inputs
with st.sidebar:
    st.header("Parameters")

    # Use session_state to manage the number of variants
    if 'num_variants' not in st.session_state:
        st.session_state.num_variants = 2

    st.number_input(
        "Number of Variants (including control)",
        min_value=2, max_value=10, step=1,
        key='num_variants',
        help="Select the total number of groups in your test, including the control."
    )
    
    st.button("Load Example Data", on_click=load_example_data, use_container_width=True)
    
    st.subheader("Test Results")
    variant_data = []
    
    # Check if example data has been loaded
    use_example = 'example_users' in st.session_state

    for i in range(st.session_state.num_variants):
        if i == 0:
            variant_name = "Control"
        else:
            variant_name = f"Variant {i}"

        st.markdown(f"**{variant_name}**")
        
        # Set default values from session state if example is loaded
        default_users = st.session_state.example_users[i] if use_example and i < len(st.session_state.example_users) else 10000
        default_conversions = st.session_state.example_conversions[i] if use_example and i < len(st.session_state.example_conversions) else int(default_users * 0.05)
        
        users = st.number_input(
            "Sample Size", min_value=1, value=default_users, step=100, 
            key=f"users_{i}", help="Total number of unique users in this variant."
        )
        conversions = st.number_input(
            "Conversions", min_value=0, max_value=users,  # Proactive validation
            value=min(default_conversions, users), step=10, 
            key=f"conv_{i}", help="Total number of unique users who converted in this variant."
        )
        variant_data.append({"name": variant_name, "users": users, "conversions": conversions})
    
    # Clear example data from state after using it once
    if use_example:
        del st.session_state.example_users
        del st.session_state.example_conversions

    st.subheader("Settings")
    credibility = st.slider(
        "Credible Interval (%)", min_value=80, max_value=99, value=95, step=1,
        help="The confidence level for the uplift's credible interval. 95% is common."
    )

    st.markdown("---")
    run_button = st.button("Run Analysis", type="primary", use_container_width=True)

# 4. Main Page for Displaying Outputs
st.markdown("---")

if run_button:
    # SRM Check
    observed_counts = [d['users'] for d in variant_data]
    if sum(observed_counts) > 0:
        chi2_stat, p_value = chisquare(f_obs=observed_counts)
        if p_value < 0.01:
            st.error("🚫 **Sample Ratio Mismatch (SRM) Detected** (p < 0.01). Results may be unreliable.")
    
    with st.spinner("Running Bayesian analysis..."):
        results_df, posteriors = run_multivariant_analysis(variant_data, credibility)
        
        st.subheader("Results Summary")
        st.dataframe(
            results_df.style.format({
                "Conversion Rate": "{:.2%}",
                "Prob. to be Best": "{:.2%}",
                "Uplift vs. Control": "{:+.2%}",
                "Credible Interval": lambda x: f"[{x[0]:.2%}, {x[1]:.2%}]"
            }).background_gradient(
                subset=["Prob. to be Best", "Uplift vs. Control"], cmap='Greens'
            )
        )

        st.subheader("Plain-Language Summary")
        best_variant = results_df.loc[results_df['Prob. to be Best'].idxmax()]
        st.success(
            f"**{best_variant['Variant']} is the most likely winner** with a "
            f"**{best_variant['Prob. to be Best']:.1%}** chance of being the best option. "
            f"It showed an average uplift of **{best_variant['Uplift vs. Control']:+.2%}** over the control."
        )

        st.subheader("Visualizations")
        variant_names = [d['name'] for d in variant_data]
        fig = plot_posteriors(posteriors, variant_names)
        st.pyplot(fig)
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Analysis', or load the example data to see how it works.")

# 5. Explanations Section
st.markdown("---")
with st.expander("ℹ️ How to interpret these results"):
    st.markdown("""
    #### Probability to be Best
    This is the key metric in a multi-variant test. It represents the probability that each variant is the single best performer out of all options, including the control. A high "Prob. to be Best" is a strong indicator of a winner.

    ---
    #### Uplift vs. Control & Credible Interval
    - **Uplift:** This shows the average estimated improvement of each variant compared **only to the control**.
    - **Credible Interval:** The range where the true uplift against the control likely falls. If this interval is entirely above zero, it's a strong sign that the variant beats the control.
    
    ---
    #### How to Make a Decision
    1.  Look for the variant with the highest **Probability to be Best**.
    2.  Check that variant's **Uplift vs. Control** and **Credible Interval** to ensure the potential gain is meaningful and you are confident it's a real improvement over the baseline.
    """)
