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


class ColumnValueSplitter:
    def __init__(self, training_category, truth_category):
        self.training_category = training_category
        self.truth_category = truth_category

    def split(self, table, metadata):
        """
        Yields
        ------
        label : str
            Name for the split
        training_table : BIOMTable
            Data for fitting detection model
        train_ids : set(str)
            A collection of train ids
        test_table : BIOMTable
            Data for testing
        test_labels : arrary like of shape (n_samples,)
            Values should be 0 or 1. 1 is anomaly and 0 is not
        """
        # returns generator
        col_val_split = column_value_splitter(
            table, metadata, self.training_category,
        )
        for label, ids, training_table in col_val_split:
            # testing
            # removing training samples from the table with all data
            train_ids = set(training_table.ids('sample'))
            test_table = table.filter(
                train_ids,
                invert=True,
                inplace=False,
            )

            # creating labels for baseline
            non_anomaly_ids = metadata.loc[
                (metadata[self.truth_category] == label)].index
            # set not anomaly as 0 and anomaly as 1
            test_labels = [0 if id_ in non_anomaly_ids
                           else 1 for id_ in test_table.ids('sample')]

            yield label, training_table, train_ids, test_table, test_labels


class ExternalScorer:
    
    def score(self, context):
        model = context["model"]      
        test_table = context["test_table"]
        return model.score_samples(test_table)


class Benchmark:

    def __init__(self, models):
        self.models = models

    def set_splitter(self, splitter):
        self.splitter = splitter

    def set_scorer(self, scorer):
        self.scorer = scorer

    def set_context(self, **kwargs):
        self._context = kwargs

    @property
    def context(self):
        return self._context

    def benchmarking_loop(self, table, metadata):
        all_results = []
        for model_name, model_attrs in self.models.items():
            model = model_attrs['model']
            splitter = self.splitter.split(table, metadata)
            for label, training_table, train_ids, test_table, test_labels \
                    in splitter:
                score_scaler = MinMaxScaler(clip=True)
                # Training
                model.fit(training_table)

                # scoring the test samples (anomaly prediction)
                self.set_context(
                    model=model, test_table=test_table,
                )
                scores = self.scorer.score(self.context)
                scores_scaled = score_scaler.fit_transform(
                    scores.reshape(-1, 1)).flatten()

                results = dict()
                # add metadata on model training to the results ouput
                results['model_name'] = model_name
                results['category'] = label

                # add evaluation metrics to the results output
                predicted_anomaly_scores = 1 - scores_scaled
                results['roc_auc'] = roc_auc_score(test_labels,
                                                   predicted_anomaly_scores)
                results['avg_prec'] = average_precision_score(
                    test_labels, predicted_anomaly_scores)

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
