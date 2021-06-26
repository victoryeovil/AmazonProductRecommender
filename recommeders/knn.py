import pandas as pd
from surprise import KNNWithMeans
from surprise import Reader, Dataset
from collections import defaultdict




def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


data = df = pd.read_csv('../resources/dataset/amazon_reviews_us_Mobile_Electronics_v1_00.tsv', sep='\t',
                        error_bad_lines=False,
                        warn_bad_lines=False,
                        usecols=['customer_id', 'product_id', 'star_rating', 'product_title', 'review_date'])
reader = Reader(
    rating_scale=(1, 5)
)

data = Dataset.load_from_df(data[['customer_id', 'product_id', 'star_rating']], reader)

trainsetfull = data.build_full_trainset()

my_k = 15
my_min_k = 5
my_sim_option = {
    'name': 'pearson', 'user_based': False
}

algo = KNNWithMeans(
    k=my_k, min_k=my_min_k,
    sim_options=my_sim_option, verbose=True
)

algo.fit(trainsetfull)

# Step 4 - Prediction

# predictions = algo.predict(uid=uid, iid=iid, r_ui=3, verbose=True)
predictions = algo.predict(str('45901892'), str('B00LU2XR8E'))

print(predictions)

top_n = get_top_n(predictions, n=5)

for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])

