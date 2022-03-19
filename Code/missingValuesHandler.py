import matplotlib.pyplot as plt
import numpy as np
import math
import random

from config import NAN, MIN_CONFIDENCE, MIN_SUPPORT, MODELS, DATASETS, MISSING_RATIO, RESULTS_PATH
from dataLoader import DataLoader
from estimator import Estimator


from efficient_apriori import apriori
import warnings
warnings.filterwarnings("ignore")


# Add missing value to  sample dataset with specific ratio
def add_missing_value(data, ratio):
    n = math.ceil(ratio * len(data))
    samples = random.sample(range(len(data)), n)
    for index in samples:
        i = random.randint(0, len(data[index]) - 1)
        data[index][i] = NAN

    return data

# https://en.wikipedia.org/wiki/Jaccard_index
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union


def plot_diffrences(title, groups_labels, list1, list1_label, list2, list2_label, ylabel = 'Jaccard Scores', save=None):
    x = np.arange(len(groups_labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, list1, width, label=list1_label)
    rects2 = ax.bar(x + width / 2, list2, width, label=list2_label)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x, groups_labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    fig.autofmt_xdate()
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

    if save:
        plt.savefig(RESULTS_PATH.format(save), bbox_inches='tight')
    else:
        plt.show()


def get_apriori_rules(transactions):
    _, rules = apriori(transactions, min_support = MIN_SUPPORT, min_confidence= MIN_CONFIDENCE, output_transaction_ids=False)
    return rules


def analayze_score_by_dataset(score, dataset_name, save):
    dataset_scores = score[dataset_name]
    labels = []
    estimate_scores = []
    missing_scores = []
    for model_name, model_scores in dataset_scores.items():
        labels.append(model_name)
        estimate_scores.append(model_scores['estimated_values_score'])
        missing_scores.append(model_scores['missing_values_score'])

    plot_diffrences(
        title=f'{dataset_name} : Scores by Model',
        groups_labels=labels,
        list1=estimate_scores,
        list1_label='Estimated Values Jaccard Score',
        list2=missing_scores,
        list2_label='Missing Values Jaccard Score',
        save=f"{dataset_name}_by_model.png" if save else None
    )



def analayze_score_by_model(score, model_name, save):
    general_model_scores = {}
    for dataset_name, dataset_scores in score.items():
        for model, model_scores in dataset_scores.items():
            if model == model_name:
                general_model_scores[dataset_name] = model_scores

    labels = []
    estimate_scores = []
    missing_scores = []
    for dataset_name, model_scores in general_model_scores.items():
        labels.append(dataset_name)
        estimate_scores.append(model_scores['estimated_values_score'])
        missing_scores.append(model_scores['missing_values_score'])

    plot_diffrences(
        title=f'{model_name} : Scores by Dataset',
        groups_labels=labels,
        list1=estimate_scores,
        list1_label='Estimated Values Jaccard Score',
        list2=missing_scores,
        list2_label='Missing Values Jaccard Score',
        save=f"{model_name}_by_dataset.png" if save else None
    )

def analayze_score(score, save=False):
    for dataset_name in DATASETS.keys():
        analayze_score_by_dataset(score, dataset_name, save)
    for _, model_name in MODELS.items():
        analayze_score_by_model(score, model_name, save)


class MissingValuesHandler:
    def __init__(self):
        pass

    def load_data(self):
        self.dataLoader = DataLoader()
        self.datasets = self.dataLoader.get_dataset()
        print (f"Data loaded succusfully for {list(self.datasets.keys())}")

    def run_all_datasets(self):
        scores = {}
        for dataset_name in DATASETS.keys():
            models_score = self.run_dataset(dataset_name)
            scores[dataset_name] = models_score
        self.all_scores = scores

    def get_all_scores(self):
        return self.all_scores

    def get_datasets(self):
        return self.datasets

    # Run the algorithm for estimating
    def run_dataset(self, dataset_name):
        data = self.datasets[dataset_name]['data']
        transactions = self.datasets[dataset_name]['transactions']
        attributes = self.datasets[dataset_name]['attributes']

        models_score = {}
        full_rules = get_apriori_rules(transactions)

        # Make the incomplete database (add missing values with the configured ratio)
        incomplete_data = add_missing_value(data=data.copy(), ratio=MISSING_RATIO)

        transactions = DataLoader.data_to_transactions(incomplete_data, attributes)
        missing_rules = get_apriori_rules(transactions)

        for model, model_name in MODELS.items():
            est = Estimator(incomplete_data)
            corrected_data = est.estimate_missing_value(model)

            transactions = DataLoader.data_to_transactions(corrected_data, attributes)
            estimated_rules = get_apriori_rules(transactions)

            # print(
            #     f"diff {round(jaccard_similarity(full_rules, missing_rules) * 100, 2)}% VS {round(jaccard_similarity(full_rules, estimated_rules) * 100, 2)}%")
            # print(
            #     f"COMPARE : {round(jaccard_similarity(full_rules, estimated_rules) * 100 - jaccard_similarity(full_rules, missing_rules) * 100, 2)}% \t\t {len(full_rules)}, {len(missing_rules)} {len(estimated_rules)}")
            models_score[model_name] = {
                'estimated_values_score': round(jaccard_similarity(full_rules, estimated_rules) * 100, 2),
                'missing_values_score': round(jaccard_similarity(full_rules, missing_rules) * 100, 2)
            }

        return models_score