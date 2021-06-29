from pathlib import Path

from surprise import Dataset, Reader, KNNWithMeans
from surprise.dump import *

from utils.loader import get_main_dataframe

# Set Dataset path
filepath = Path().joinpath('../', 'resources', 'dataset', 'amazon_reviews_us_Digital_Software_v1_00.tsv')
data_main = get_main_dataframe(filepath)

# Parse the Pandas Dataframe to Surprise readable format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_main[["customer_id", "product_id", "star_rating"]], reader)

# Generate Training Dataset
trainingSet = data.build_full_trainset()

# To use item-based cosine similarity
sim_options = {
    "name": "cosine",
    "user_based": False,  # Compute  similarities between items
}
# Train and Test KNN Model with Training/ Test Dataset
algo = KNNWithMeans(k=30, min_k=5, sim_options=sim_options)
algo.fit(trainingSet)
predictions = algo.test(trainingSet.build_testset())

# Save the trained KNN Model
filepath = Path().joinpath('../', 'resources', 'models', 'knn.model')
dump(file_name=filepath, predictions=predictions, algo=algo, verbose=1)
