# SlamDunker

This project is meant to figure out if it is possible to detect troll tweets purely based on the content of the tweet.

We started off with a pretrained DistilBERT model from Huggingface and refined the model with our dataset. In total, we trained the model for over 6 hours on Kaggle's TPU service.

![Webpage](./demo.png)

## Running the project
- Download the model [here](https://drive.google.com/file/d/1xXIiFc-eTr8E3PxZ6BW1KpJ5Rw38MXJw/view?usp=share_link). Store it in `/site/model.bin`.
- Make sure you have installed [Poetry](https://python-poetry.org/docs/)
- Enter the `site` folder and run `poetry install`
- Start up the server by running `poetry run flask --app run run`
- Visit the website at `localhost:5000`

## Training the model
You can find our training code in `/kaggle_notebook` and our preprocessing in `/notebooks`.
If you upload the notebook to kaggle, along with the data, you can train the model using their TPU service,.