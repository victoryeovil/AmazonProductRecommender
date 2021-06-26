import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans
from surprise.dump import *

data_main = pd.read_csv('../resources/dataset/amazon_reviews_us_Digital_Software_v1_00.tsv',
                        sep='\t',
                        error_bad_lines=False,
                        warn_bad_lines=False)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_main[["customer_id", "product_id", "star_rating"]], reader)

trainingSet = data.build_full_trainset()

# To use item-based cosine similarity
sim_options = {
    "name": "cosine",
    "user_based": False,  # Compute  similarities between items
}
algo = KNNWithMeans(k=30, min_k=5, sim_options=sim_options)
algo.fit(trainingSet)

predictions = algo.test(trainingSet.build_testset())

filename = '../resources/models/knn.model'

dump(file_name=filename, predictions=predictions, algo=algo, verbose=1)
