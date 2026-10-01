"""Checks for leakage boundaries, weighting, missing data and metric correctness."""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
from evaluate_drug_attributes import (
    annotation_metrics, equal_compound_weights, predict_labels,
    regression_metrics, split_data, transform, weighted_ap, aggregate,
    run_regression, run_replicates, TARGETS,
)


class AttributeTests(unittest.TestCase):
    def test_group_and_batch_boundaries(self):
        meta = pd.DataFrame({'compound_id': list('aabbccddeeff'),
                             'batch': [0, 1, 0, 2, 1, 2, 0, 1, 1, 2, 0, 2]})
        chem = dict(zip('abcdef', ['A', 'A', 'B', 'B', 'C', 'C']))
        for protocol in ['held_out_compound', 'held_out_batch', 'held_out_chemical_group']:
            d, _, units, splits = split_data(meta, {'x': np.ones((12, 2))}, protocol, chem, 3)
            for train, test in splits:
                self.assertFalse(set(d.compound_id.iloc[train]) & set(d.compound_id.iloc[test]))
                if protocol == 'held_out_batch':
                    self.assertFalse(set(d.batch.iloc[train]) & set(d.batch.iloc[test]))
                if protocol == 'held_out_chemical_group':
                    self.assertFalse(set(units[train]) & set(units[test]))

    def test_unresolved_chemical_compounds_excluded(self):
        meta = pd.DataFrame({'compound_id': list('abcx'), 'batch': 0})
        d, x, _, _ = split_data(meta, {'x': np.arange(4)[:, None]},
                               'held_out_chemical_group', dict(zip('abc', 'ABC')), 3)
        self.assertEqual(d.compound_id.tolist(), list('abc'))
        np.testing.assert_array_equal(x['x'][:, 0], [0, 1, 2])

    def test_preprocessing_train_only_and_empty_column(self):
        train = np.array([[0., np.nan], [2., np.nan]])
        a, b = transform(train, np.array([[100., 7.]]))
        np.testing.assert_allclose(a[:, 0], [-1, 1])
        self.assertEqual(b[0, 0], 99)
        self.assertEqual(b.shape, (1, 2))
        self.assertTrue(np.isfinite(b).all())

    def test_equal_compound_weight(self):
        ids = np.array(['a', 'a', 'a', 'b'])
        w = equal_compound_weights(ids)
        self.assertAlmostEqual(w[:3].sum(), w[3])

    def test_ap_matches_sklearn_with_ties_and_zero_weights(self):
        y = np.array([1, 0, 1, 0, 1])
        score = np.array([.4, .4, .1, .8, .8])
        weights = np.array([[1, 1, 1, 1, 1], [0, 2, 1, 3, 2]])
        actual = weighted_ap(y, score, weights)
        for i, w in enumerate(weights):
            self.assertAlmostEqual(actual[i], average_precision_score(y, score, sample_weight=w))

    def test_bootstrap_identical_prediction_baseline_delta_zero(self):
        y = np.array([0, 1, 0, 1])
        pred = np.array([.1, .9, .4, .7])
        result = annotation_metrics(y, pred, pred, np.array(['a', 'a', 'b', 'b']), 100)
        self.assertEqual(result['ap_delta'], 0)
        self.assertEqual(result['ap_delta_ci_low'], 0)
        self.assertEqual(result['ap_delta_ci_high'], 0)

    def test_constant_label_fold(self):
        a = np.array([[0.], [1.], [2.]])
        y = np.array([[0, 1], [0, 1], [0, 1]])
        scores = predict_labels(a, y, np.array([[3.]]), np.ones(3), False)
        np.testing.assert_array_equal(scores, [[0, 1]])

    def test_parallel_label_fits_match_serial(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(20, 4))
        y = (rng.normal(size=(20, 3)) > 0).astype(int)
        serial = predict_labels(x, y, x[:5], np.ones(20), False, workers=1)
        parallel = predict_labels(x, y, x[:5], np.ones(20), False, workers=3)
        np.testing.assert_allclose(serial, parallel)

    def test_regression_perfect_and_constant_targets(self):
        frame = pd.DataFrame(dict(compound_id=list('abcd'), bootstrap_unit=list('abcd'),
                                  truth=[1., 2., 3., 4.], prediction=[1., 2., 3., 4.], baseline=2.5))
        result = regression_metrics(frame, 20)
        self.assertAlmostEqual(result['r2'], 1)
        self.assertAlmostEqual(result['mse_skill'], 1)
        frame['truth'] = 1.
        self.assertTrue(np.isnan(regression_metrics(frame, 0)['r2']))

    def test_multioutput_regression_with_one_missing_target(self):
        rng = np.random.default_rng(1)
        meta = pd.DataFrame(dict(compound_id=[f'c{i}' for i in range(10)],
                                 batch=np.arange(10) % 2, compound_concentration_um=1.))
        for target in TARGETS:
            meta[target] = rng.normal(size=10)
        meta.loc[0, TARGETS[-1]] = np.nan
        arrays = {'pca_normalized': rng.normal(size=(10, 3))}
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), protocols=['held_out_compound'],
                                   models=['linear'], bootstrap=10)
            run_regression(meta, arrays, args, {})
            result = pd.read_csv(args.output / 'continuous_scores.csv')
            self.assertEqual(len(result), 3 * len(TARGETS))
            self.assertTrue((result.loc[result.target == TARGETS[-1], 'n_profiles'] == 9).all())

    def test_aggregation_preserves_missing_dose_and_row_alignment(self):
        meta = pd.DataFrame(dict(compound_id=['b', 'a', 'a'], batch=[0, 0, 0],
                                 compound_concentration_um=[np.nan, 1., 1.],
                                 compound_pathway=['B', 'A', 'A'], compound_target=['T', 'T', 'T']))
        for target in TARGETS:
            meta[target] = [5., 2., 4.]
        result, arrays = aggregate(meta, {'x': np.array([[5.], [2.], [4.]])},
                                   ['compound_id', 'compound_concentration_um', 'batch'])
        self.assertEqual(result.compound_id.tolist(), ['a', 'b'])
        self.assertTrue(np.isnan(result.compound_concentration_um.iloc[1]))
        np.testing.assert_array_equal(arrays['x'][:, 0], [3., 5.])

    def test_retrieval_excludes_same_plate_and_uses_matched_queries(self):
        meta = pd.DataFrame(dict(compound_id=['a', 'a', 'b', 'b'],
                                 compound_concentration_um=1., plate=[0, 1, 0, 1],
                                 batch=[0, 1, 0, 1], well_id=['a0', 'a1', 'b0', 'b1']))
        vectors = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), max_queries=3)
            run_replicates(meta, {'pca_normalized': vectors, 'brightfield': vectors}, args)
            result = pd.read_csv(args.output / 'replicate_queries.csv')
            self.assertTrue((result.n_gallery == 2).all())
            self.assertTrue((result.hit_at_1 == 1).all())
            for pool in result.pool.unique():
                a = result[(result.pool == pool) & (result.feature == 'pca_normalized')].query_well.tolist()
                b = result[(result.pool == pool) & (result.feature == 'brightfield')].query_well.tolist()
                self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
