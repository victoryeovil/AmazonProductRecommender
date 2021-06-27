import heapq
from collections import defaultdict
from operator import itemgetter

from surprise import Dataset, Reader
from surprise.dump import *

from utils.common import get_customer_reviewed_products, get_product_name


def get_knn_recommendation(customer_id, data_main, top_n=10):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data_main[["customer_id", "product_id", "star_rating"]], reader)

    products = data_main[['product_id', 'product_title']].drop_duplicates()

    training_set = data.build_full_trainset()

    filename = 'resources/models/knn.model'

    model = load(file_name=filename)
    similarity_matrix = model[1].compute_similarities()

    test_subject = customer_id
    k = top_n

    test_subject_iid = training_set.to_inner_uid(test_subject)
    test_subject_ratings = training_set.ur[test_subject_iid]
    k_neighbors = heapq.nlargest(k, test_subject_ratings, key=lambda t: t[1])

    candidates = defaultdict(float)

    for itemID, rating in k_neighbors:
        try:
            similarities = similarity_matrix[itemID]
            for innerID, score in enumerate(similarities):
                candidates[innerID] += score * (rating / 5.0)
        except:
            continue

    watched = {}
    for itemID, rating in training_set.ur[test_subject_iid]:
        watched[itemID] = 1

    recommendations = []

    position = 0
    for itemID, rating_sum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if not itemID in watched:
            recommendations.append(get_product_name(products, training_set.to_raw_iid(itemID)))
            position += 1
            if position > top_n:
                break  # We only want top 10

    rated_products = get_customer_reviewed_products(data_main, customer_id)

    return [rated_products], recommendations
