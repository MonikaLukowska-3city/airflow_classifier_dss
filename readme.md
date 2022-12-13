# Implementation of ML models with Airflow
<img src="https://dssconf.pl/i/dss-logo.e139193a.svg"/>


### Installation

1. docker-compose up
2. Launch Airflow
3. Install MS SQL and run the environment.

---

### General Information

The project is created for the conference [Data Science Summit 2022](https://dssconf.pl/). The project shows the workflow from data loading to validation, prediction execution, and a generalized training scheme. Data from . Subsequent batches of data are generated.

---

### Usage

After using the docker-compose up command and running airflow lockalhost:8000, create the dss database and several tables according to the init.sql code
Then you can go to Airflow. The first is the DAG that initializes the model, and the next is the DAG that performs the prediction on new batches of data. You run them in this order, one after another.

After the data validation task is performed, predictor histograms will be created in the project's /results/data_analysis folder. Models will be learned from task "x" and saved in the project's /models folder.
Model metadata, model quality assessment results, and classifier probability deciles will be saved in the /results folder of the project.

---

### Workflow

1. Downloading data from ms sql.
2. Data validation and saving results to the dss database.
3. Loading the current champion model.
4. Uploading the champion model, description of the prediction and training process + saving the results to sql. (The results of probability deciles, accuracy, roc_auc, feature importance, precision and sensitivity attributes are saved to the database.)
     - Model initiation (training the first champion model on a batch of data 1. Having real y and predicted y after production, accuracy and roc_auc assessment is performed. Saving the results on the database.
     - Having a champion model after training on batch 1, we perform prediction using this model on batch 2. For this batch of data, we know the actual and predicted values, we evaluate score accurcy and roc_auc. We assess accuracy and check if the condition roc_auc >= 0.5 has been met:
         - If the accuracy assessment of the current champion learned on batch 1 is maintained, then the current champion remains the prediction champion for subsequent batches of data and there is no challenger.
         - If the accuracy assessment of the current champion learned on batch 1 is worse than the accuracy of the prediction on batch 2, it goes to the model training path -> i.e. a new challenger model is taught. The data from the previous batches + the current one (in this case batch 1 +2) are used to teach the challenger. After training the challenger, the results of both models, i.e. the current champion and the new challenger, are stored in the database.
     - On the next data batch (in this case, data batch 3), the prediction of the current champion and challenger is performed. Only now can a fair evaluation of the models be made on a clean dataset for these two models, neither of which was trained on this batch of data.
     - Accuracy and roc_auc are counted for champion and challenger:
         - If the accuracy is higher for the challenger, he becomes a new champion and there is currently no challenger for him.
         - If the accuracy is lower for the challenger, the current champion becomes the champion.
     - Depending on the end of the path from point 5.1. and 5.2. a prediction is made on the next game 4 for the champion and the points cycle above is repeated.
     - In a situation where the trained model is worse than the champion twice in a row, we have a warning that the modeled phenomenon should be analyzed and other steps to improve modeling should be implemented.
