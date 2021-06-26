import pandas as pd


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


def load_customer_ids(path_to_customers):

    """Load movie titles from database records.

    Parameters
    ----------
    path_to_customers : str
        Relative or absolute path to customer database stored
        in .tsv format.

    Returns
    -------
    list[str]
        Customer ID.
    """
    df = pd.read_csv(path_to_customers, sep='\t',
                     error_bad_lines=False,
                     warn_bad_lines=False, )
    df = df.dropna()
    customer_list = df['customer_id'].drop_duplicates().to_list()
    return customer_list

