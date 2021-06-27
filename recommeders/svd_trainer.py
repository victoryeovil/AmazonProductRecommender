import pandas as pd
from surprise import Dataset, Reader, SVDpp
from surprise.dump import *

data_main = pd.read_csv('../resources/dataset/amazon_reviews_us_Digital_Software_v1_00.tsv',
                        sep='\t',
                        error_bad_lines=False,
                        warn_bad_lines=False)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_main[["customer_id", "product_id", "star_rating"]], reader)

trainingSet = data.build_full_trainset()


algo = SVDpp()
algo.fit(trainingSet)

predictions = algo.test(trainingSet.build_testset())

filename = '../resources/models/svd.model'

dump(file_name=filename, predictions=predictions, algo=algo, verbose=1)
