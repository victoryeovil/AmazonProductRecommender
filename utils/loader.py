from os import walk

import pandas as pd


# @st.cache
def load_product_titles(path_to_products):
    """Load product titles from database records.

    Parameters
    ----------
    path_to_products : str
        Relative or absolute path to product database stored
        in .tsv format.

    Returns
    -------
    list[str]
        Product titles.

    """
    df = pd.read_csv(path_to_products, sep='\t',
                     error_bad_lines=False,
                     warn_bad_lines=False, )
    df = df.dropna()
    product_list = df['product_title'].to_list()
    return product_list


# @st.cache
def load_customer_ids(data_main):
    """Load Customer IDs from Dataframe.

    Parameters
    ----------
    data_main : Dataframe
        Relative or absolute path to customer database stored
        in .tsv format.

    Returns
    -------
    list[str]
        Customer ID.
    """
    data_main = data_main.dropna()
    customer_list = data_main['customer_id'].drop_duplicates().to_list()

    return customer_list


def get_trained_models(path_to_models):
    return next(walk(path_to_models), (None, None, []))[2]  # [] if no file


def get_datasets(path_to_datasets):
    return next(walk(path_to_datasets), (None, None, []))[2]  # [] if no file


def get_main_dataframe(dataset_path):
    data_main = pd.read_csv(dataset_path,
                            sep='\t',
                            error_bad_lines=False,
                            warn_bad_lines=False)
    # Drop Null Values
    combine_product_rating = data_main.dropna(axis=0, subset=['product_title'])

    # Get Rating Count Per Product
    product_ratingCount = (combine_product_rating.
        groupby(by=['product_title'])['star_rating'].
        count().
        reset_index().
        rename(columns={'star_rating': 'totalRatingCount'})
    [['product_title', 'totalRatingCount']]
        )

    # Combine Rating count to get record
    rating_with_totalRatingCount = combine_product_rating.merge(product_ratingCount, left_on='product_title',
                                                                right_on='product_title', how='left')
    # Set threshold for required review count
    popularity_threshold = 50

    # Get records with product popularity threshold
    cleaned_data = rating_with_totalRatingCount.query(
        "totalRatingCount >= @popularity_threshold")

    return cleaned_data
