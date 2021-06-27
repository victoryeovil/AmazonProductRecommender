# Simple Recommender System Using Scikit Surprise

## Prequisites
- Ensure you have Python v3.6 or later
- Install the required dependencies present in the `requirements.txt`

## How To Setup Project
- Download the [Digital_Software_V1](https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Software_v1_00.tsv.gz) Dataset
- Place the dataset `amazon_reviews_us_Digital_Software_v1_00.tsv` in the `resources > dataset` directory
- Navigate inside the`recommenders` directory.
- Run the `svd_trainer.py` & `knn_trainer.py` files to train the required Models. (This process will be improved later).
- The models should be present in the `resources > models` directory.
- Once the model has been generated, return to the root of the Project.
- Execute the following command to run the web application:
```bash
$ streamlit run main.py 
```
- The application should be accessible via the url `http://localhost:8501`