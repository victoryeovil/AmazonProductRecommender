from pathlib import Path

from surprise import Dataset, Reader, SVD
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

# Train and Test KNN Model with Training/ Test Dataset
algo = SVD(n_epochs=100, lr_all=0.1, random_state=5)
algo.fit(trainingSet)
predictions = algo.test(trainingSet.build_testset())

# Save trained SVD Model
filepath = Path().joinpath('../', 'resources', 'models', 'svd.model')
dump(file_name=filepath, predictions=predictions, algo=algo, verbose=1)
