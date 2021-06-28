import heapq
from pathlib import Path

from surprise import Dataset, Reader
from surprise.dump import *

from utils.common import get_top_n, get_customer_reviewed_products, get_product_name


def get_svd_recommendation(customer_id, data_main, top_n=10):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data_main[["customer_id", "product_id", "star_rating"]], reader)

    products = data_main[['product_id', 'product_title']].drop_duplicates()

    training_set = data.build_full_trainset()

    filename = Path().joinpath('resources', 'models', 'svd.model')

    model = load(file_name=filename)

    test_subject_iid = training_set.to_inner_uid(customer_id)
    test_subject_ratings = training_set.ur[test_subject_iid]

    k = top_n + len(test_subject_ratings)

    watched = {}
    for itemID, rating in training_set.ur[test_subject_iid]:
        watched[training_set.to_raw_iid(itemID)] = 1

    all_recommendations = []
    for item in products.product_id:
        all_recommendations.append(model[1].predict(uid=customer_id, iid=item))

    k_neighbors = heapq.nlargest(k, all_recommendations, key=lambda t: t[3])
    top_t = get_top_n(k_neighbors, n=k)

    rated_products = get_customer_reviewed_products(data_main, customer_id)

    position = 0
    recommendations = []
    for rec in top_t[customer_id]:
        if not rec[0] in watched:
            recommendations.append(get_product_name(products, rec[0]))
            position += 1
            if position > top_n:
                break  # We only want top 10

    return [rated_products], recommendations
