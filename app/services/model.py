import pyodbc
import os
from dao.customer_data_dao import CustomerDataDao
from dao.models_dao import ModelsDao
from dao.metadata_dao import MetadataDao
from dao.pdp_deciles_dao import PdpDecilesDao
from dao.results_dao import ResultsDao

import logging
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score,roc_auc_score
import pickle
import numpy as np
from datetime import datetime


logging.basicConfig(filename='/results/output.log', level=logging.DEBUG)

class ModelService:
    def __init__(self):
        DATABASE_NAME = os.getenv('DATABASE_NAME')
        DATABASE_SERVER = os.getenv('DATABASE_SERVER')
        DATABASE_LOGIN = os.getenv('DATABASE_LOGIN')
        DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD')
        DATABASE_PORT = os.getenv('DATABASE_PORT')
        con_string = "DRIVER={ODBC Driver 18 for SQL Server};SERVER=" + DATABASE_SERVER +  "," + DATABASE_PORT +  ";Encrypt=no;DATABASE="+  DATABASE_NAME + ";UID=" + DATABASE_LOGIN +  ";PWD=" + DATABASE_PASSWORD
        self.connSqlServer = pyodbc.connect(con_string)

        self.models_dao = ModelsDao(self.connSqlServer)
        self.pdp_deciles_dao = PdpDecilesDao(self.connSqlServer)
        self.meta_dao = MetadataDao(self.connSqlServer)
        self.results_dao = ResultsDao(self.connSqlServer)
        self.customer_data_dao = CustomerDataDao(self.connSqlServer)

        self.features = ['fea_1',
            'fea_2',
            'fea_3',
            'fea_4', 
            'fea_5', 
            'fea_6', 
            'fea_7', 
            'fea_8', 
            'fea_9', 
            'fea_10', 
            'fea_11']
            
        self.MIN_ROC_AUC = 0.5


    def load_data_to_predict(self, part_id, use_historical = False):
        logging.info(f"call load_data_to_predict => (part_id: {part_id}, use_historical: {use_historical})")
        data_df = self.customer_data_dao.load(part_id, use_historical)
        data_df.fillna(0, inplace=True)
        return data_df



    def predict(self, model_id, data_df):
        logging.info(f"call predict => model_id: {model_id}")

        if model_id is not None:
            model_path = self.models_dao.get_model_path(model_id)
        else:
            model_path, model_id = self.models_dao.get_champion_model_path_and_id()


        with open(model_path, 'rb') as pickle_file:
            model = pickle.load(pickle_file)

            y_pred = model.predict(data_df[self.features])
            y_prob_positive = model.predict_proba(data_df[self.features])[:, 1]
            y_prob_negative = model.predict_proba(data_df[self.features])[:, 0]

            data_df['prediction'] = y_pred
            data_df['pdp_1'] = y_prob_positive
            data_df['pdp_0'] = y_prob_negative

            self.results_dao.save(model_id, data_df)

            output_dir = model_path.replace("/models/", "").replace("/model.sav", "")
            precision_score, recall_score, roc_auc_score, acc_score = self.__score(model_id, output_dir, data_df)

            challenger_id = self.models_dao.get_challenger_model(model_id)

            result = {
                "used_model_id": model_id,
                "challenger_id": challenger_id,
                "model_path": model_path,
                "precision_score": precision_score,
                "recall_score": recall_score,
                "roc_auc_score": roc_auc_score,
                "acc_score": acc_score,
                "predictedRecords": len(data_df.index),
                "status": "ok"
            }

        return result


    def get_last_processed_part(self):
        part = self.results_dao.get_last_processed_part()
        result = {
            "status": "ok",
            "last_part": part,
            "next_part": part + 1
        }
        return result


    def get_champion_model_id(self):
        model_path, model_id = self.models_dao.get_champion_model_path_and_id()
        return model_id


    def rate_model(self, model_id):
        acc, previous_acc = self.models_dao.get_model_acc(model_id)
        need_to_train = acc + 0.04 < previous_acc

        roc_auc = self.models_dao.get_model_roc_auc(model_id)
        is_fail = roc_auc < self.MIN_ROC_AUC
        fail_count = self.models_dao.model_fails(model_id, is_fail)

        result = {
            "status": "ok",
            "acc": acc,
            "previous_acc": previous_acc,
            "need_to_train": need_to_train,
            "fail_count": fail_count
        }
        return result


    def evaluate_models(self):
        champion_id, challenger_id = self.models_dao.get_champion_and_challenger_models()
        champion_fail_count = self.models_dao.model_fails(champion_id, False)

        champion_acc, champion_previous_acc = self.models_dao.get_model_acc(champion_id)
        challenger_acc, challenger_previous_acc = self.models_dao.get_model_acc(challenger_id)
        challenger_roc_auc = self.models_dao.get_model_roc_auc(challenger_id)

        #test if challenger_id is better?
        challenger_win = challenger_acc > champion_acc and challenger_roc_auc >= self.MIN_ROC_AUC
       
        if challenger_win:
            self.models_dao.set_champion_model(challenger_id)
        else:
            self.models_dao.set_challenger_model(champion_id, None)


        result = {
            "status": "ok",
            "champion_acc": champion_acc,
            "challenger_acc": challenger_acc,
            "challenger_roc_auc": challenger_roc_auc, 
            "challenger_win": challenger_win,
            "champion_id": champion_id,
            "challenger_id": challenger_id,
            "champion_fail_count": champion_fail_count
        }
        return result

    
    def valid_data(self, part_id):
        now = datetime.now()
        dir = now.strftime("%m%d%Y_%H%M%S")
        output_dir = f"/data_analysis/{part_id}_{dir}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        #Fetch data to analyse
        features_df = self.customer_data_dao.load(part_id)[self.features]
        self.__daw_histogram(features_df, output_dir)
        
        result = {
            "status": "ok",
            "analyzed_part": part_id,
            "output_dir": output_dir
        }
        return result


    def initial_train(self):
        logging.info(f"call initial_train()")

        #Build model and return it's X_test set
        result = self.train(part_id=1)

        #Mark model as champion
        self.models_dao.set_champion_model(result["model_id"])
       
        return  result



    def train(self, part_id):
        logging.info(f"train => (part_id: {part_id})")
        now = datetime.now()
        output_dir = now.strftime("%m%d%Y_%H%M%S")
        if not os.path.exists(f"/models/{output_dir}"):
            os.makedirs(f"/models/{output_dir}")
            os.makedirs(f"/results/{output_dir}")

        #Fetch data from parts <= part_id with label
        data_df = self.customer_data_dao.load_with_labels(part_id)

        y = data_df["label"]
        x = data_df.copy()
        x.drop(columns=["label"], inplace=True)
        x.fillna(0, inplace=True)

        #Splita data set to test and train
        ros = RandomOverSampler()
        X_train_res, y_train_res = ros.fit_resample(x, y)
        X_train, X_test, y_train, y_test = train_test_split(X_train_res, y_train_res, train_size = 0.80, test_size = 0.20, random_state = 1)

        logging.info(f"y_train: {y_train.head()}")
        logging.info(f"y_test: {y_test.head()}")


        #Build model
        model = tree.DecisionTreeClassifier(max_depth=5, criterion= "entropy")
        model.fit(X_train[self.features], y_train)

        # save the model to disk
        filename = f'/models/{output_dir}/model.sav'
        pickle.dump(model, open(filename, 'wb'))

        # Train results....
        y_train_pred = model.predict(X_train[self.features])

        conf_matrix = confusion_matrix(y_train, y_train_pred.round())
        self.__draw_conf_matrix(conf_matrix,output_dir)

        path = model.cost_complexity_pruning_path(X_train[self.features],y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        self.__draw_total_impurity_effective_alpha(ccp_alphas,impurities,output_dir)


        importance = model.feature_importances_
        self.__draw_importance(importance, output_dir)
        self.__draw_tree(model, output_dir)
        logging.info(f"Feature importance {importance}")

        models = []
        for ccp_alpha in ccp_alphas:
            model = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            model.fit(X_train[self.features], y_train)
            models.append(model)
        logging.info("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(models[-1].tree_.node_count, ccp_alphas[-1]))

        models = models[:-1]
        ccp_alphas = ccp_alphas[:-1]

        node_counts = [model.tree_.node_count for model in models]
        depth = [model.tree_.max_depth for model in models]
        self.__draw_number_nodes_alpha(ccp_alphas, node_counts, depth,output_dir)


        train_scores = [model.score(X_train[self.features], y_train) for model in models]
        test_scores = [model.score(X_test[self.features], y_test) for model in models]
        self.__draw_accuracy_alpha(ccp_alphas, train_scores, test_scores,output_dir)
        

        model_id = self.models_dao.save(filename, ccp_alphas[-1], models[-1].tree_.node_count)
        self.meta_dao.save(model_id, importance)

        #Calculate prediction result for X_test to get model score
        self.predict(model_id, X_test)

        # update challenger for champion
        champion_path, champion_id = self.models_dao.get_champion_model_path_and_id()
        if champion_id is not None:
            self.models_dao.set_challenger_model(champion_id, model_id)


        result = {
            "model_id": model_id,
            "champion_id": champion_id,
            "status": "ok"
        }
        return result


    def __daw_histogram(self,features_df,output_path):
        for feature in self.features:
            features_df[feature].plot.hist(grid=True, bins=20, rwidth=0.9,color='#607c8e')
            plt.title(f'Histogram of {feature}')
            plt.xlabel(f'{feature}')
            plt.ylabel('count')
            plt.grid(axis='y', alpha=0.75)
            plt.savefig(f"{output_path}/histogram_{feature}.png")

    def __draw_number_nodes_alpha(self, ccp_alphas, node_counts, depth,output_dir):
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
        ax[0].set_xlabel("alpha")
        ax[0].set_ylabel("number of nodes")
        ax[0].set_title("Number of nodes vs alpha")
        ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
        ax[1].set_xlabel("alpha")
        ax[1].set_ylabel("depth of tree")
        ax[1].set_title("Depth vs alpha")
        fig.tight_layout()
        fig.savefig(f"/results/{output_dir}/number_nodes_alpha.png")


    def __draw_accuracy_alpha(self,ccp_alphas, train_scores, test_scores,output_dir):
        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs alpha for training and testing sets")
        ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
        ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
        ax.legend()
        fig.savefig(f"/results/{output_dir}/accuracy_alpha.png")


    def __draw_total_impurity_effective_alpha(self,ccp_alphas,impurities,output_dir):
        fig, ax = plt.subplots()
        ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
        ax.set_xlabel("effective alpha")
        ax.set_ylabel("total impurity of leaves")
        ax.set_title("Total Impurity vs effective alpha for training set")
        fig.savefig(f"/results/{output_dir}/total_impurity_effective_alpha.png")


    def __draw_importance(self,importance,output_dir):
        plt.figure(figsize=(20,12))
        plt.bar([x for x in range(len(importance))], importance)
        plt.savefig(f"/results/{output_dir}/importance.png")


    def __draw_tree(self,model,output_dir):
        fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (20,20), dpi=600)
        tree.plot_tree(model, fontsize=11)
        fig.savefig(f'/results/{output_dir}/tree.png')


    def __draw_conf_matrix(self,conf_matrix, output_dir):
        LABELS = ['0', '1']
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
        plt.title("Confusion Matrix")
        plt.ylabel('actual class')
        plt.xlabel('predicted class')
        plt.savefig(f"/results/{output_dir}/confusion_matrix.png")
    

    def __score(self, model_id, output_dir, data_df):
        now = datetime.now()
        stamp = now.strftime("%m%d%Y_%H%M%S")

        logging.info(f"call score => (model_id: {model_id}), stamp: {stamp}")

        #Get all data predicted with this model with real label's
        data_df = self.results_dao.get_results_with_labels(model_id, data_df)
        logging.info(f"score data_df: {data_df.head()}")

        y_label = data_df['label']
        y_pred = data_df['prediction']

        precision_score_val = np.round((precision_score(y_label, y_pred)), 4)
        recall_score_val = np.round((recall_score(y_label, y_pred)), 4)
        roc_auc_score_val = np.round((roc_auc_score(y_label, y_pred)), 4)
        acc_val = np.round((accuracy_score(y_label, y_pred)), 4)


        logging.info(f'Precision score: {precision_score_val}')
        logging.info(f'Recall score: {recall_score_val}')
        logging.info(f'Roc auc score: {roc_auc_score_val}')
        logging.info(f'ACC score: {acc_val}')


        self.models_dao.save_score(model_id, precision_score_val, recall_score_val, roc_auc_score_val, acc_val)

        y_prob_positive = data_df['pdp_1']
        y_prob_positive.to_csv(f'/results/{output_dir}/prob_{stamp}.csv')
        deciles = np.round(np.percentile(y_prob_positive, np.arange(0, 100, 10)), 5)
        logging.info(f'Deciles of pdp data: {deciles}')

        self.pdp_deciles_dao.save(model_id, deciles)

        return precision_score_val, recall_score_val, roc_auc_score_val, acc_val
