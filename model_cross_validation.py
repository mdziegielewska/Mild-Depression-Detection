from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import numpy as np


class ModelCrossValidation:
    
    def __init__(
        self,
        model,
        train_data,
        train_true_results,
        test_data,
        test_true_results,
        fold_number
    ):
        self.model = model
        self.train_data = train_data
        self.fold_number = fold_number
        self.test_data = test_data

        self.test_true_results = test_true_results
        self.train_true_results = train_true_results

        self.scoring = ["balanced_accuracy", "roc_auc", "precision", "recall", "f1"]
        self.test_balanced_accuracy_list = []
        self.test_roc_auc_list = []
        self.test_f1_score_list = []
        self.test_recall_list = []
        self.test_precision_list = []

    def main_cycle(self):
        cv_results = cross_validate(
            self.model,
            self.train_data,
            self.train_true_results,
            cv=self.fold_number,
            return_train_score=True,
            scoring=self.scoring,
            return_estimator=True,
        )

        for cross_validation_model in cv_results["estimator"]:
            self.gather_test_results(cross_validation_model)

    def gather_test_results(self, model):
        predictions = model.predict(self.test_data)

        test_balanced_accuracy = balanced_accuracy_score(
            self.test_true_results, predictions
        )
        self.test_balanced_accuracy_list.append(test_balanced_accuracy)

        test_roc_auc = roc_auc_score(self.test_true_results, model.predict_proba(self.test_data)[::, 1])
        self.test_roc_auc_list.append(test_roc_auc)

        test_precision = precision_score(
            self.test_true_results, predictions, average="weighted"
        )
        self.test_precision_list.append(test_precision)

        test_recall = recall_score(
            self.test_true_results, predictions, average="weighted"
        )
        self.test_recall_list.append(test_recall)

        test_f1_score = f1_score(
            self.test_true_results, predictions, average="weighted"
        )
        self.test_f1_score_list.append(test_f1_score)

    def display_test_results(self):
        print("Test balanced accuracy", np.mean(self.test_balanced_accuracy_list))
        print()

        print("Test roc auc", np.mean(self.test_roc_auc_list))
        print()

        print("Test precision", np.mean(self.test_precision_list))
        print()

        print("Test recall", np.mean(self.test_recall_list))
        print()

        print("Test f1 score", np.mean(self.test_f1_score_list))

    @staticmethod
    def display_metrics(cv_results):
        for value in cv_results.keys():
            str_value = str(value)
            if str_value.startswith("test_"):
                str_value = str_value.replace("test", "Validation")
                str_value = str_value.replace("_", " ")
                print(str_value, cv_results[value].mean())
            if str_value.startswith("train_"):
                str_value = str_value.replace("train", "Training")
                str_value = str_value.replace("_", " ")
                print(str_value, cv_results[value].mean())
                print()