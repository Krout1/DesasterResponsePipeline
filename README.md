# Disaster Response Pipeline Project
[Figure Eight](https://www.figure-eight.com/) provided data related to messages, which are categorized into different classifications. The messages appeared durind desasters.
This project tries to make an connection of the messages to these categories. Hence, a quicker response time to desasters would be possible
Using machine learning techniques, we shold be able to predict the category.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
