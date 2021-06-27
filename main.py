import streamlit as st

from recommeders.knn_tester import get_knn_recommendation
from recommeders.svd_tester import get_svd_recommendation
from utils.loader import load_customer_ids, get_trained_models

# Data handling dependencies

# Data Loading
# title_list = load_product_titles('resources/dataset/amazon_reviews_us_Digital_Software_v1_00.tsv')

customer_list = load_customer_ids('resources/dataset/amazon_reviews_us_Digital_Software_v1_00.tsv')


def main():

    page_options = ["Recommender System", "Model Trainer"]

    page_selection = st.sidebar.selectbox("Choose Option", page_options)

    if page_selection == "Recommender System":
        # Header contents
        st.write('# Product Recommender Engine')
        st.write('### EXPLORE CS412 Unsupervised Predictions')
        st.image('resources/images/amazon-recommends.png', use_column_width=True)

        st.write('### Recommendation For Customers')
        selected_customer = st.selectbox('Customer ID', customer_list[1400:15700])

        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('K Nearest Neighbours',
                        'Singular Value Decomposition'))

        if sys == 'K Nearest Neighbours':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        previous_likes, top_recommendations = get_knn_recommendation(customer_id=selected_customer,
                                                                                     top_n=5)
                    st.title("The Customer Previously Reviewed:")
                    for rec in previous_likes:
                        st.write(rec)

                    st.title("We think the customer will like:")
                    for i, rec in enumerate(top_recommendations):
                        st.write(str(i + 1) + '. ' + rec)
                except:

                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")
        else:
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        previous_likes, top_recommendations = get_svd_recommendation(customer_id=selected_customer,
                                                                                     top_n=5)
                    st.title("The Customer Previously Reviewed:")
                    for rec in previous_likes:
                        st.write(rec)

                    st.title("We think the customer will like:")
                    for i, rec in enumerate(top_recommendations):
                        st.write(str(i + 1) + '. ' + rec)
                except:

                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")

    else:
        models = get_trained_models('resources/models/')
        st.write('### Available Models')
        selected_models = st.selectbox('Selected Model', models)

        datasets = get_trained_models('resources/dataset/')
        st.write('### Available Datasets')
        selected_models = st.selectbox('Selected Dataset', datasets)


if __name__ == '__main__':
    main()
