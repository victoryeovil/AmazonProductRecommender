from pathlib import Path

import streamlit as st

from recommeders.knn_recommender import get_knn_recommendation
from recommeders.svd_recommender import get_svd_recommendation
from utils.loader import load_customer_ids, get_trained_models, get_main_dataframe

# Data Loading
data_main = get_main_dataframe(
    Path().joinpath(
        "resources", "dataset", "amazon_reviews_us_Digital_Software_v1_00.tsv"
    )
)
customer_list = load_customer_ids(data_main)


def main():
    page_options = ["Recommender System", "Model Trainer"]

    page_selection = st.sidebar.selectbox("Choose Option", page_options)

    if page_selection == "Recommender System":
        # Header contents
        st.write("# Product Recommender Engine")

        st.write("## Based On Amazon Customers Reviews")
        selected_customer = st.selectbox("Customer ID", customer_list[1400:15700])

        # Recommender System algorithm selection
        sys = st.radio(
            "Select an algorithm",
            ("K Nearest Neighbours", "Singular Value Decomposition"),
        )

        if st.button("Recommend"):
            with st.spinner("Crunching the numbers..."):
                try:
                    _get_model_recommendations(sys, selected_customer)
                except:

                    st.error(
                        "Oops! Looks like this algorithm doesn't work.\
                                We'll need to fix it!"
                    )

    else:
        _get_resource_selectbox("models", "### Available Models", "Selected Model")
        _get_resource_selectbox("dataset", "### Available Datasets", "Selected Dataset")


def _get_model_recommendations(sys, selected_customer):
    if sys == "K Nearest Neighbours":
        # Get Recommendations from the KNN trained model using Cosine Similarity
        (time_taken, previous_likes, top_recommendations,) = get_knn_recommendation(
            customer_id=selected_customer, data_main=data_main, top_n=5
        )

    else:
        # Get Recommendations from the SVD trained model
        (time_taken, previous_likes, top_recommendations,) = get_svd_recommendation(
            customer_id=selected_customer, data_main=data_main, top_n=5
        )
    st.write("### The Customer Previously Reviewed:")
    for rec in previous_likes:
        st.write(rec)

    st.write("### We think the customer will like:")
    for i, rec in enumerate(top_recommendations):
        st.write(str(i + 1) + ". " + rec)

    st.write(f"_Recommendations Generated in {time_taken} seconds_")


def _get_resource_selectbox(arg0, arg1, arg2):
    models = get_trained_models(Path().joinpath("resources", arg0))
    st.write(arg1)
    selected_models = st.selectbox(arg2, models)


if __name__ == "__main__":
    main()
