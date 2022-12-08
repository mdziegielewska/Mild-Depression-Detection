from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class ModelSingleTrain:

    def __init__(self, model, train_data, train_true_results, test_data, test_true_results, seed):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.test_true_results = test_true_results
        self.train_true_results = train_true_results
        self.seed = seed

        self.train_data, self.validation_data,\
            self.train_true_results, self.validation_true_results = train_test_split(self.train_data,
                                                                                     self.train_true_results,
                                                                                     test_size=0.2,
                                                                                     shuffle=True,
                                                                                     random_state=self.seed,
                                                                                     stratify=self.train_true_results)

    def main_cycle(self):
        self.model.fit(self.train_data, self.train_true_results)

        self._display_metrics(self.train_data,
                              self.train_true_results,
                              "Train")
        self._display_metrics(self.validation_data,
                              self.validation_true_results,
                              "Validation")
        self._display_metrics(self.test_data,
                              self.test_true_results,
                              "Test")

    def display_confusion_matrix(self):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
        axes[0].title.set_text("Validation confusion matrix")
        axes[1].title.set_text("Test confusion matrix")
        ConfusionMatrixDisplay.from_estimator(self.model,
                                              self.validation_data,
                                              self.validation_true_results,
                                              ax=axes[0],
                                              colorbar=False)
        ConfusionMatrixDisplay.from_estimator(self.model,
                                              self.test_data,
                                              self.test_true_results,
                                              ax=axes[1],
                                              colorbar=False)
        plt.show()

    def _display_metrics(self, data, classes, set_name):
        predictions = self.model.predict(data)

        balanced_accuracy = balanced_accuracy_score(classes,
                                                    predictions)

        precision = precision_score(classes,
                                    predictions,
                                    average="weighted")

        recall = recall_score(classes,
                              predictions,
                              average="weighted")

        f1 = f1_score(classes,
                      predictions,
                      average="weighted")

        print()
        print(f"{set_name} balanced accuracy {balanced_accuracy}")
        print()
        print(f"{set_name} precision {precision}")
        print()
        print(f"{set_name} recall {recall}")
        print()
        print(f"{set_name} f1 score {f1}")
        print()
    
    def train_and_get_metrics(self):
        self.model.fit(self.train_data, self.train_true_results)
        data = self.test_data
        classes = self.test_true_results
        predictions = self.model.predict(data)
        cm = confusion_matrix(classes, predictions, labels=[0, 1])
        balanced_accuracy = balanced_accuracy_score(classes,
                                                    predictions)

        precision = precision_score(classes,
                                    predictions,
                                    average="weighted")

        recall = recall_score(classes,
                              predictions,
                              average="weighted")

        f1 = f1_score(classes,
                      predictions,
                      average="weighted")

        return (balanced_accuracy, precision, recall, f1, cm)
    
    def show_cm(self, cm):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot()
