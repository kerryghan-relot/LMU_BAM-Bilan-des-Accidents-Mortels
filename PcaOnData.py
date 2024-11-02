# Importer les bibliothèques
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from time import perf_counter
from sklearn.svm import *



class PcaOnData:
    """
    À l'initialisation, cela peut prendre un peu de temps, c'est normal puisqu'on calcul l'accuracy du classifier donné en paramètre.

    :param X: Your data
    :param y: Your classes
    :param random_state: By default, it is 42
    :param test_size: By default, 30% of the samples are taken for test
    :param classifier: The given classifier must implement a .fit() and .predict() method, by default it is Random Forest
    """
    def __init__(self, X, y, random_state: int = 42, test_size: float = 0.3, classifier = RandomForestClassifier):
        self.original_data = X
        self.orignal_classes = y
        self.PCA_object = PCA(random_state=random_state)
        self.reduced_data = self.PCA_object.fit_transform(self.original_data)
        self.random_state = random_state
        self.test_size = test_size
        self.classifier = classifier(random_state=random_state)
        self.classifier_name = classifier.__name__
        self.original_accuracy = self._compute_accuracy(X)
        self.n_features = self.original_data.shape[1]

    def _compute_accuracy(self, data) -> float:
        """
        Calcul l'accuracy des données fournies en entrées en utilisant les paramètres initialisés

        :param data: Les données dont on veut calculer l'accuracy
        :return: L'accuracy comprise entre 0 et 1 sous forme de float
        """
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            data,
            self.orignal_classes,
            test_size=self.test_size,
            random_state=self.random_state
        )
        # Modèle Défini en paramètre du constructeur
        self.classifier.fit(X_train, y_train)
        # Prédictions sur l'ensemble de test
        y_pred = self.classifier.predict(X_test)
        # Calcul de l'accuracy
        return accuracy_score(y_test, y_pred)

    def component_vs_variance(self) -> List[float]:
        """
        Calcul la somme des variances pour tous les n premiers axes et enregistre la figure dans le fichier "ComponentVsVariance.png"

        :return: La liste des variances pour les n n-premiers axes
        """
        # Compute the running sum
        variance_ratios = [e for e in self.PCA_object.explained_variance_ratio_]
        for i in range(1, self.n_features):
            variance_ratios[i] += variance_ratios[i - 1]

        # Plot
        plt.plot(variance_ratios)
        plt.title("Number of components vs. Explained Variance Ratio")
        plt.ylabel("Explained Variance Ratio")
        plt.xlabel("Number of Components")
        plt.savefig("ComponentVsVariance.png")

        # Return the variance ratios
        return variance_ratios

    def component_vs_accuracy(self, debug: bool = False, limit: int = None) -> List[float]:
        """
        WARNING: Cette fonction est assez longue à éxécuter
        Elle calcul l'accuracy du classifier choisi pour toutes les valeurs de n, en prenant les n premiers axes de l'ACP.

        :param debug:
        :param limit:
        :return:
        """
        # Set default value to `self.n_features` if not already set
        limit = limit or self.n_features

        # Start timing the process
        start = perf_counter()
        accuracies = []
        for i in range(1, limit):
            accuracies.append( self._compute_accuracy(self.reduced_data[:, :i]) )
        end = perf_counter()
        duration = end - start

        if debug:
            print(f"It took {duration:.2f} seconds to run this motherfucker")

        # Plot
        plt.plot(accuracies)
        plt.suptitle(f"{self.classifier_name} Accuracy with different number of components ")
        plt.title("red line being the accuracy of the orignal data", color='grey')
        plt.ylabel("N first axis after PCA")
        plt.xlabel("Accuracy")
        plt.axhline(self.original_accuracy, color='r')
        plt.text(-3.5, self.original_accuracy, "Original\nAccuracy",
                 horizontalalignment='center', verticalalignment='center', weight='bold', style="italic")
        plt.savefig("ComponentVsAccuracy.png")

        return accuracies