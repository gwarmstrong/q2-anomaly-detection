from sklearn.metrics import roc_auc_score, average_precision_score
from q2_anomaly_detection.utils import as_dense


def benchmarking_loop(models, table, metadata, truth_category, splitter):
    all_results = []
    for model_name, model_attrs in models.items():
        model = model_attrs['model']
        score_scaler = model_attrs['scale_score']
        for val, ids, training_table in splitter():
            # Training
            model.fit(as_dense(training_table))

            # testing
            # removing training samples from the table with all data
            train_ids = set(training_table.ids('sample'))
            test_table = table.filter(train_ids, invert=True, inplace=False)

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
            results['avg_prec'] = average_precision_score(true_labels,
                                                          predicted_anomaly_scores)

            # assigning individual anomaly scores to samples in the results
            # output
            results['anomaly_scores'] = [{'sample_id': id_, 'score': score,
                                          'scaled_score': scaled,
                                          'train_test': 'train' if id_ in
                                                                   train_ids
                                          else 'test',
                                          } for
                                         id_, score, scaled in zip(
                    test_table.ids('sample'),
                    scores, scores_scaled,
                )
                                         ]

            all_results.append(results)

    return all_results
