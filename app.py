import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import beta, chisquare
# Altair and Matplotlib are now imported only when needed

# 1. Set Page Configuration
st.set_page_config(
    page_title="Uplift Estimator | Bayesian Toolkit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Core Calculation Functions ---
@st.cache_data(persist="disk")
def run_multivariant_analysis(variant_data, credibility):
    """
    Performs Bayesian analysis for multiple variants using vectorized operations.
    """
    alpha_prior, beta_prior = 1, 1
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

    stacked_samples = posterior_samples
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
    posteriors = [beta(a, b) for a, b in zip(alpha_posts, beta_posts)]
    
    return results_df, posteriors

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
    # UPDATED: Lazy-load Altair for faster initial load
    import altair as alt

    def plot_altair_charts(posteriors, results_df):
        # 1. Prepare data for the posterior density plot
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

        # 2. Create the posterior density plot
        posterior_chart = alt.Chart(plot_df).mark_area(opacity=0.6).encode(
            x=alt.X('Conversion Rate:Q', axis=alt.Axis(format='%', title='Conversion Rate')),
            y=alt.Y('Density:Q', title='Density'),
            color=alt.Color('Variant:N', scale=alt.Scale(scheme='tableau10'), title="Variant"),
            tooltip=[
                alt.Tooltip('Variant:N'),
                alt.Tooltip('Conversion Rate:Q', format='.3%'),
            ]
        ).properties(
            title="Posterior Distributions"
        ).interactive()

        # 3. Create the "Probability to be Best" bar chart
        prob_best_chart = alt.Chart(results_df).mark_bar().encode(
            x=alt.X('Prob. to be Best:Q', axis=alt.Axis(format='%'), title="Probability to be Best"),
            y=alt.Y('Variant:N', sort='-x', title=None),
            color=alt.Color('Variant:N', scale=alt.Scale(scheme='tableau10'), legend=None),
            tooltip=[alt.Tooltip('Variant:N'), alt.Tooltip('Prob. to be Best:Q', format='.2%')]
        ).properties(
            title="Chance to be the Best Variant"
        )
        
        text = prob_best_chart.mark_text(align='left', baseline='middle', dx=4).encode(
            text=alt.Text('Prob. to be Best:Q', format=".2%")
        )

        # 4. Combine the charts side-by-side
        combined_chart = alt.hconcat(
            posterior_chart, 
            (prob_best_chart + text).properties(width=300)
        ).resolve_scale(
            color='independent'
        )
        
        return combined_chart

    if any(d['conversions'] > d['users'] for d in variant_data):
        st.error("Conversions cannot exceed the sample size for a variant.")
    else:
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
            best_variant_row = results_df.loc[results_df['Prob. to be Best'].idxmax()]
            
            prob_best = best_variant_row['Prob. to be Best']
            ci = best_variant_row['Credible Interval']
            best_variant_name = best_variant_row['Variant']

            if prob_best >= prob_threshold and ci[0] > 0:
                st.success(
                    f"✅ **{best_variant_name} is a clear winner.** "
                    f"It has a high **{prob_best:.2%}** chance of being the best (above your {prob_threshold:.0%} threshold), and its credible interval "
                    f"**[{ci[0]:.2%}, {ci[1]:.2%}]** is entirely above zero, indicating a reliable positive uplift."
                )
            elif prob_best >= prob_threshold and ci[0] <= 0:
                 st.warning(
                    f"⚠️ **{best_variant_name} is the most likely winner, but the result is not conclusive.** "
                    f"While it has a strong **{prob_best:.2%}** chance of being the best, its credible interval "
                    f"**[{ci[0]:.2%}, {ci[1]:.2%}]** still overlaps with zero. This means we can't be certain about the size of the uplift."
                )
            else:
                st.info(
                    f"ℹ️ **The test is inconclusive.** No variant reached your **{prob_threshold:.0%}** threshold for being the best. "
                    f"While **{best_variant_name}** performed best in this test, its **{prob_best:.2%}** "
                    f"chance of being truly best is not high enough to declare a confident winner."
                )

            st.subheader("Visualizations")
            chart = plot_altair_charts(posteriors, results_df)
            st.altair_chart(chart, use_container_width=True)
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
    1.  Look for the variant with the highest **Probability to be Best**. Use the slider in the sidebar to set your threshold for declaring a winner (95% is a common choice).
    2.  Check that variant's **Credible Interval**. If it is entirely above zero, you can be confident that the uplift is positive.
    """)
