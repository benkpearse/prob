import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import beta, chisquare
# Altair is now imported only when needed

# 1. Set Page Configuration
st.set_page_config(
    page_title="Uplift Estimator | Bayesian Toolkit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Core Calculation Functions ---
@st.cache_data(persist="disk")
def run_multivariant_analysis(variant_data_tuple, credibility, alpha_prior, beta_prior):
    """
    Performs Bayesian analysis for multiple variants using vectorized operations.
    """
    variant_data = [{'name': v[0], 'users': v[1], 'conversions': v[2]} for v in variant_data_tuple]
    
    samples = 30000
    num_variants = len(variant_data)
    
    conversions = np.array([d['conversions'] for d in variant_data])
    users = np.array([d['users'] for d in variant_data])
    
    alpha_posts = alpha_prior + conversions
    beta_posts = beta_prior + users - conversions

    rng = np.random.default_rng(seed=42)
    
    posterior_samples = beta.rvs(
        alpha_posts, 
        beta_posts, 
        size=(samples, num_variants), 
        random_state=rng
    ).T

    best_sample_rates = np.max(posterior_samples, axis=0)
    loss_samples = best_sample_rates - posterior_samples
    expected_loss = np.mean(loss_samples, axis=1)

    best_variant_indices = np.argmax(posterior_samples, axis=0)
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
            "Expected Loss": expected_loss[i],
            "Uplift vs. Control": mean_uplift,
            "Credible Interval": (ci_lower, ci_upper)
        })

    results_df = pd.DataFrame(results)
    posteriors = [beta(a, b) for a, b in zip(alpha_posts, beta_posts)]
    
    return results_df, posteriors

# --- Helper Functions for UI ---
def load_example_data():
    st.session_state.num_variants = 3
    st.session_state.example_users = [10000, 10000, 10000]
    st.session_state.example_conversions = [500, 550, 520]

def reset_inputs():
    """Clears session state to reset the form."""
    keys_to_delete = ['num_variants', 'example_users', 'example_conversions']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    # Clear individual variant inputs if they exist
    for i in range(10): # Max variants
        if f"users_{i}" in st.session_state:
            del st.session_state[f"users_{i}"]
        if f"conv_{i}" in st.session_state:
            del st.session_state[f"conv_{i}"]


# 2. Page Title and Introduction
st.title("📈 Multi-Variant Uplift Estimator")
st.markdown("This tool interprets A/B/n test results using Bayesian inference to find the best performing variant.")

# 3. Sidebar for All User Inputs
with st.sidebar:
    st.header("Parameters")

    if 'num_variants' not in st.session_state:
        st.session_state.num_variants = 2

    st.number_input(
        "Number of Variants (including control)",
        min_value=2, max_value=10, step=1,
        key='num_variants',
        help="Select the total number of groups in your test, including the control."
    )
    
    # --- NEW: Reset and Load Example Buttons ---
    col1, col2 = st.columns(2)
    with col1:
        st.button("Load Example", on_click=load_example_data, use_container_width=True)
    with col2:
        st.button("Reset Inputs", on_click=reset_inputs, use_container_width=True)

    st.subheader("Test Results")
    variant_data = []
    
    use_example = 'example_users' in st.session_state

    for i in range(st.session_state.num_variants):
        if i == 0:
            variant_name = "Control"
        else:
            variant_name = f"Variant {i}"

        st.markdown(f"**{variant_name}**")
        
        default_users = st.session_state.example_users[i] if use_example and i < len(st.session_state.example_users) else 10000
        default_conversions = st.session_state.example_conversions[i] if use_example and i < len(st.session_state.example_conversions) else int(default_users * 0.05)
        
        users = st.number_input(
            "Sample Size", min_value=1, value=default_users, step=100, 
            key=f"users_{i}", help="Total number of unique users in this variant."
        )
        conversions = st.number_input(
            "Conversions", min_value=0, max_value=users,
            value=min(default_conversions, users), step=10, 
            key=f"conv_{i}", help="Total number of unique users who converted in this variant."
        )
        variant_data.append({"name": variant_name, "users": users, "conversions": conversions})
    
    if use_example:
        del st.session_state.example_users
        del st.session_state.example_conversions

    st.subheader("Prior Beliefs")
    prior_mode = st.radio(
        "Choose your prior",
        ["Uninformative (Default)", "Informative (from data)"],
        horizontal=True,
        help="Use 'Uninformative' if you have no past data. Use 'Informative' to incorporate historical knowledge."
    )
    if prior_mode == "Informative (from data)":
        hist_conv = st.number_input("Historical Conversions", min_value=0, value=100, step=1)
        hist_users = st.number_input("Historical Users", min_value=1, value=2000, step=1)
        alpha_prior = hist_conv
        beta_prior = hist_users - hist_conv
    else:
        alpha_prior = 1
        beta_prior = 1

    st.subheader("Settings")
    
    prob_threshold = st.slider(
        "Probability to be Best Threshold (%)",
        min_value=80, max_value=99, value=95, step=1,
        help="The 'Probability to be Best' a variant must exceed to be considered a winner."
    ) / 100.0
    
    credibility = st.slider(
        "Credible Interval (%)", min_value=80, max_value=99, value=95, step=1,
        help="The confidence level for the uplift's credible interval. 95% is common."
    )

    st.markdown("---")
    run_button = st.button("Run Analysis", type="primary", use_container_width=True)

# 4. Main Page for Displaying Outputs
st.markdown("---")

if run_button:
    import altair as alt

    def plot_posterior_chart(posteriors, results_df):
        plot_data = []
        min_x = min(p.ppf(0.0001) for p in posteriors)
        max_x = max(p.ppf(0.9999) for p in posteriors)
        x_zoom_range = np.linspace(min_x, max_x, 300)

        for i, post in enumerate(posteriors):
            variant_name = results_df['Variant'].iloc[i]
            density = post.pdf(x_zoom_range)
            for x, y in zip(x_zoom_range, density):
                plot_data.append({"Variant": variant_name, "Conversion Rate": x, "Density": y})
        
        plot_df = pd.DataFrame(plot_data)

        posterior_chart = alt.Chart(plot_df).mark_area(opacity=0.6).encode(
            x=alt.X('Conversion Rate:Q', axis=alt.Axis(format='%', title='Conversion Rate')),
            y=alt.Y('Density:Q', title='Density'),
            color=alt.Color('Variant:N', scale=alt.Scale(scheme='tableau10'), title="Variant"),
            tooltip=[alt.Tooltip('Variant:N'), alt.Tooltip('Conversion Rate:Q', format='.3%')]
        ).properties(
            title="Posterior Distributions of Conversion Rates"
        ).interactive()
        
        return posterior_chart

    if any(d['conversions'] > d['users'] for d in variant_data):
        st.error("Conversions cannot exceed the sample size for a variant.")
    else:
        observed_counts = [d['users'] for d in variant_data]
        if sum(observed_counts) > 0:
            chi2_stat, p_value = chisquare(f_obs=observed_counts)
            if p_value < 0.01:
                st.error("🚫 **Sample Ratio Mismatch (SRM) Detected** (p < 0.01). Results may be unreliable.")
        
        with st.spinner("Running Bayesian analysis..."):
            variant_data_tuple = tuple((d['name'], d['users'], d['conversions']) for d in variant_data)
            results_df, posteriors = run_multivariant_analysis(variant_data_tuple, credibility, alpha_prior, beta_prior)
            
            st.subheader("Results Summary")
            # --- NEW: Visually grouped metrics explainer ---
            with st.container(border=True):
                st.markdown("##### Key Metrics Explained")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("**Prob. to be Best** (?)", help="The chance that each variant is the single best performer.")
                with col2:
                    st.markdown("**Expected Loss** (?)", help="The average amount you 'lose' by choosing this variant instead of the true best one. Lower is better.")
                with col3:
                    st.markdown("**Uplift vs. Control** (?)", help="The average estimated improvement compared only to the control.")
                with col4:
                    st.markdown("**Credible Interval** (?)", help="The range where the true uplift against the control likely falls.")

            display_df = results_df.copy()
            display_df['Credible Interval'] = display_df['Credible Interval'].apply(
                lambda x: f"[{x[0]:.2%}, {x[1]:.2%}]"
            )

            st.dataframe(
                display_df.style.format({
                    "Conversion Rate": "{:.2%}",
                    "Prob. to be Best": "{:.2%}",
                    "Expected Loss": "{:.4%}",
                    "Uplift vs. Control": "{:+.2%}",
                }).background_gradient(
                    subset=["Prob. to be Best", "Uplift vs. Control"], cmap='Greens'
                ).background_gradient(
                    subset=["Expected Loss"], cmap='Reds'
                )
            )

            st.subheader("Test Outcome")
            best_variant_row = results_df.loc[results_df['Prob. to be Best'].idxmax()]
            
            prob_best = best_variant_row['Prob. to be Best']
            ci = best_variant_row['Credible Interval']
            best_variant_name = best_variant_row['Variant']

            if best_variant_name != "Control" and prob_best >= prob_threshold and ci[0] > 0:
                st.success(
                    f"✅ **Outcome: Clear Winner.** {best_variant_name} is a clear winner because its "
                    f"**{prob_best:.2%}** chance of being the best is above your **{prob_threshold:.0%}** threshold, "
                    f"and its credible interval **[{ci[0]:.2%}, {ci[1]:.2%}]** is entirely positive."
                )
            elif best_variant_name != "Control" and prob_best >= prob_threshold and ci[0] <= 0:
                 st.warning(
                    f"⚠️ **Outcome: Likely Winner, but Risk Remains.** {best_variant_name} is the most likely winner, as its "
                    f"**{prob_best:.2%}** chance of being best is above your **{prob_threshold:.0%}** threshold. "
                    f"However, its credible interval **[{ci[0]:.2%}, {ci[1]:.2%}]** still includes zero, indicating a risk of a neutral or negative outcome."
                )
            elif best_variant_name == "Control" and prob_best >= prob_threshold:
                variants_only_df = results_df[results_df['Variant'] != 'Control']
                if not variants_only_df.empty:
                    top_variant_row = variants_only_df.loc[variants_only_df['Uplift vs. Control'].idxmax()]
                    top_variant_name = top_variant_row['Variant']
                    top_variant_ci = top_variant_row['Credible Interval']
                    st.error(
                        f"❌ **Outcome: Clear Loser.** The **Control** is the best performing option with a **{prob_best:.2%}** chance of being the best. "
                        f"The top variant, **{top_variant_name}**, showed a credible interval of **[{top_variant_ci[0]:.2%}, {top_variant_ci[1]:.2%}]**, "
                        "indicating it likely performs worse than the control."
                    )
                else:
                    st.error(f"❌ **Outcome: Clear Loser.** The **Control** was the best performing option.")
            else:
                st.info(
                    f"ℹ️ **Outcome: Inconclusive.** The test is inconclusive because no variant (including the Control) reached your "
                    f"**{prob_threshold:.0%}** threshold for being the best. While **{best_variant_name}** performed best, "
                    f"there is not enough evidence to declare a confident winner."
                )

            st.subheader("Visualizations")
            st.markdown(
                "**Posterior Distributions** (?)",
                help="This chart shows our belief about the true conversion rate for each variant after seeing the data. Look for separation between the curves—the less they overlap, the more certain we are that a real difference exists."
            )
            chart = plot_posterior_chart(posteriors, results_df)
            st.altair_chart(chart, use_container_width=True)
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Analysis', or load the example data to see how it works.")

# 5. Explanations Section
st.markdown("---")
with st.expander("ℹ️ About the Methodology"):
    st.markdown("""
    #### The Key Metrics Explained in Detail
    
    **1. Probability to be Best**
    - **What it is:** The probability that each variant is the single best performer out of all options. It is the primary metric for making a decision in a multi-variant test.
    - **How to use it:** Look for the variant with the highest probability. If this value is above the decision threshold you set in the sidebar (e.g., 95%), you have a confident winner.
    
    **2. Expected Loss (or Regret)**
    - **What it is:** A risk metric. It quantifies the average amount of conversion rate you might "lose" by choosing a specific variant if it isn't actually the best one. It's the opportunity cost of making the wrong decision.
    - **How to use it:** The variant with the **lowest** expected loss is the safest, most robust choice. This is especially useful when the "Probability to be Best" is close between two variants. A low expected loss means the decision is low-risk.
    
    **3. Uplift vs. Control**
    - **What it is:** The average estimated improvement of each variant when compared **only to the control group**.
    - **How to use it:** This metric tells you the magnitude of the improvement over your baseline. It helps you understand if the winning variant's effect is large enough to be meaningful for your business.
    
    **4. Credible Interval**
    - **What it is:** The range where we are confident the true uplift against the control lies. For example, a 95% credible interval of `[1%, 5%]` means we are 95% certain the true uplift is between 1% and 5%.
    - **How to use it:** Check if the interval is entirely above zero. If it is, you can be confident that the variant has a positive effect. If it includes zero, you cannot rule out the possibility that the variant has no effect or even a negative one.

    ---
    #### The Visualization: Posterior Distributions
    The chart provides a visual confirmation of the summary table. It shows our updated belief about the true conversion rate for each variant after seeing the data. The curve that is furthest to the right belongs to the likely winning variant. **Look for separation between the curves**—the less they overlap, the more certain we are that a real difference exists.
    """)
