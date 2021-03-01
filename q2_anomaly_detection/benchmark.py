import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from q2_anomaly_detection.utils import as_dense
from q2_anomaly_detection.cross_validation import column_value_splitter
from scipy.stats import rankdata


class Results(list):

    def __init__(self, results_dict):
        super().__init__(results_dict)

    def long_form(self):
        res_df = pd.DataFrame.from_dict(
            {(e['model_name'], e['category'], score['sample_id']):
                [
                    score['score'],
                    score['scaled_score'],
                    score['score_rank'],
                    score['train_test']]
             for e in self
             for score in e['anomaly_scores']
             },
            orient='index',
            columns=['anomaly_score',
                     'scaled_score',
                     'score_rank',
                     'train_test'],
        )
        res_df.index = pd.MultiIndex.from_tuples(res_df.index,
                                                 names=['model_name',
                                                        'category',
                                                        'sample_id'])
        return res_df

    def short_form(self):
        agg_results = pd.DataFrame(self).drop('anomaly_scores', axis=1)
        return agg_results


class Benchmark:

    def __init__(self, models):
        self.models = models

    @staticmethod
    def splitter(table, metadata, training_category):
        return column_value_splitter(table, metadata, training_category)

    def benchmarking_loop(self, table, metadata, truth_category,
                          training_category):
        all_results = []
        for model_name, model_attrs in self.models.items():
            model = model_attrs['model']
            splitter = self.splitter(table, metadata, training_category)
            for val, ids, training_table in splitter:
                score_scaler = MinMaxScaler(clip=True)
                # Training
                model.fit(as_dense(training_table))

                # testing
                # removing training samples from the table with all data
                train_ids = set(training_table.ids('sample'))
                test_table = table.filter(
                    train_ids,
                    invert=True,
                    inplace=False,
                )

                # scoring the test samples (anomaly prediction)
                scores = model.score_samples(as_dense(test_table))
                scores_scaled = score_scaler.fit_transform(
                    scores.reshape(-1, 1)).flatten()

                # creating labels for baseline
                non_anomaly_ids = metadata.loc[
                    (metadata[truth_category] == val)].index
                # set not anomaly as 0 and anomaly as 1
                true_labels = [0 if id_ in non_anomaly_ids
                               else 1 for id_ in test_table.ids('sample')]

                results = dict()
                # add metadata on model training to the results ouput
                results['model_name'] = model_name
                results['category'] = val

                # add evaluation metrics to the results output
                predicted_anomaly_scores = 1 - scores_scaled
                results['roc_auc'] = roc_auc_score(true_labels,
                                                   predicted_anomaly_scores)
                results['avg_prec'] = average_precision_score(
                    true_labels, predicted_anomaly_scores)

                # assigning individual anomaly scores to samples in the results
                # output
                ranked_scores = rankdata(scores)
                results['anomaly_scores'] = [
                    {'sample_id': id_, 'score': score,
                     'scaled_score': scaled,
                     'score_rank': ranked_score,
                     'train_test': 'train' if id_ in train_ids else 'test'} for
                    id_, score, scaled, ranked_score in zip(
                        test_table.ids('sample'),
                        scores, scores_scaled, ranked_scores
                    )
                ]

                all_results.append(results)

        return Results(all_results)
