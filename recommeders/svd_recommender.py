import heapq
import time
from pathlib import Path

from surprise import Dataset, Reader
from surprise.dump import *

from utils.common import get_top_n, get_customer_reviewed_products, get_product_name


def get_svd_recommendation(customer_id, data_main, top_n=10):
    initial = time.perf_counter()

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data_main[["customer_id", "product_id", "star_rating"]], reader)
    products = data_main[['product_id', 'product_title']].drop_duplicates()
    training_set = data.build_full_trainset()

    # Read saved Model
    filename = Path().joinpath('resources', 'models', 'svd.model')
    model = load(file_name=filename)

    # Convert Raw Id to Inner Id for selected Customer
    customer_inner_id = training_set.to_inner_uid(customer_id)
    selected_customer_ratings = training_set.ur[customer_inner_id]

    k = top_n + len(selected_customer_ratings)

    # Add Items Already Review To A Dictionary
    rated = {}
    for itemID, rating in training_set.ur[customer_inner_id]:
        rated[training_set.to_raw_iid(itemID)] = 1

    # Get Predictions for All Products for selected Customer
    all_recommendations = []
    for item in products.product_id:
        all_recommendations.append(model[1].predict(uid=customer_id, iid=item))

    # Sort Predicted Ratings in Descending Order & get the top_n + customer review count
    k_recommendations = heapq.nlargest(k, all_recommendations, key=lambda t: t[3])
    top_t = get_top_n(k_recommendations, n=k)

    rated_products = get_customer_reviewed_products(data_main, customer_id)

    # Loop through list & Remove Already reviewed Products
    position = 0
    recommendations = []
    for rec in top_t[customer_id]:
        if not rec[0] in rated:
            recommendations.append(get_product_name(products, rec[0]))
            position += 1
            if position > top_n:
                break  # We only want top_n recommendations

    final = time.perf_counter()
    time_taken = f"{final - initial:0.4f}"

    return time_taken, [rated_products], recommendations
