import heapq
import time
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

from surprise import Dataset, Reader
from surprise.dump import *

from utils.common import get_customer_reviewed_products, get_product_name


def get_knn_recommendation(customer_id, data_main, top_n=10):
    initial = time.perf_counter()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data_main[["customer_id", "product_id", "star_rating"]], reader)

    products = data_main[['product_id', 'product_title']].drop_duplicates()

    training_set = data.build_full_trainset()

    # Read saved Model
    filename = Path().joinpath('resources', 'models', 'knn.model')
    model = load(file_name=filename)

    # Generate the Similarity Matrix
    similarity_matrix = model[1].compute_similarities()

    k = top_n

    # Convert Raw Id to Inner Id for selected Customer
    customer_inner_id = training_set.to_inner_uid(customer_id)
    selected_customer_ratings = training_set.ur[customer_inner_id]
    k_neighbors = heapq.nlargest(k, selected_customer_ratings, key=lambda t: t[1])

    candidates = defaultdict(float)

    # Get Item Similar to those reviewed by Customer
    for itemID, rating in k_neighbors:
        try:
            similarities = similarity_matrix[itemID]
            for innerID, score in enumerate(similarities):
                candidates[innerID] += score * (rating / 5.0)
        except:
            continue

    # Add Items Already Review To A Dictionary
    rated = {}
    for itemID, rating in training_set.ur[customer_inner_id]:
        rated[itemID] = 1

    # Loop through list & Remove Already reviewed Products
    recommendations = []
    position = 0
    for itemID, rating_sum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if not itemID in rated:
            recommendations.append(get_product_name(products, training_set.to_raw_iid(itemID)))
            position += 1
            if position > top_n:
                break  # We only want top_n recommendations

    rated_products = get_customer_reviewed_products(data_main, customer_id)

    final = time.perf_counter()
    time_taken = f"{final - initial:0.4f}"

    return time_taken, [rated_products], recommendations
