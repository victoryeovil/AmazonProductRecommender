from pathlib import Path

import streamlit as st

from recommeders.knn_recommender import get_knn_recommendation
from recommeders.svd_recommender import get_svd_recommendation
from utils.loader import load_customer_ids, get_trained_models, get_main_dataframe

# Data Loading
data_main = get_main_dataframe(Path().joinpath('resources', 'dataset', 'amazon_reviews_us_Digital_Software_v1_00.tsv'))
customer_list = load_customer_ids(data_main)


def main():
    page_options = ["Recommender System", "Model Trainer"]

    page_selection = st.sidebar.selectbox("Choose Option", page_options)

    if page_selection == "Recommender System":
        # Header contents
        st.write('# Product Recommender Engine')

        st.write('## Based On Amazon Customers Reviews')
        selected_customer = st.selectbox('Customer ID', customer_list[1400:15700])

        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('K Nearest Neighbours',
                        'Singular Value Decomposition'))

        if sys == 'K Nearest Neighbours':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        # Get Recommendations from the KNN trained model using Cosine Similarity
                        previous_likes, top_recommendations = get_knn_recommendation(customer_id=selected_customer,
                                                                                     data_main=data_main,
                                                                                     top_n=5)
                    st.write("## The Customer Previously Reviewed:")
                    for rec in previous_likes:
                        st.write(rec)

                    st.write("## We think the customer will like:")
                    for i, rec in enumerate(top_recommendations):
                        st.write(str(i + 1) + '. ' + rec)
                except:

                    st.error("Oops! Looks like this algorithm doesn't work.\
                              We'll need to fix it!")
        else:
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        # Get Recommendations from the SVD trained model
                        previous_likes, top_recommendations = get_svd_recommendation(customer_id=selected_customer,
                                                                                     data_main=data_main,
                                                                                     top_n=5)
                    st.write("### The Customer Previously Reviewed:")
                    for rec in previous_likes:
                        st.write(rec)

                    st.write("### We think the customer will like:")
                    for i, rec in enumerate(top_recommendations):
                        st.write(str(i + 1) + '. ' + rec)
                except:

                    st.error("Oops! Looks like this algorithm doesn't work.\
                              We'll need to fix it!")

    else:
        models = get_trained_models(Path().joinpath('resources', 'models'))
        st.write('### Available Models')
        selected_models = st.selectbox('Selected Model', models)

        datasets = get_trained_models(Path().joinpath('resources', 'dataset'))
        st.write('### Available Datasets')
        selected_models = st.selectbox('Selected Dataset', datasets)


if __name__ == '__main__':
    main()
