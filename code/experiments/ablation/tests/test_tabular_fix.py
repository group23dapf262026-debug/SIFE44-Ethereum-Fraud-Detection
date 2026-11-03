"""Protect the explicit LR conditioning change and LightGBM named-column contract."""
import copy
import tempfile
from pathlib import Path
import unittest
import warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sife20.tabular_inputs import tabular_input_frame, tabular_column_names, check_estimator_columns
from sife20.features import PURE36_INDICES
from sife20.protocol import open_study
from sife20.training import make_estimator
from test_study import fixture

class TabularFixChecks(unittest.TestCase):

    def test_lightgbm_named_fit_predict_without_feature_name_warning(self):
        rng = np.random.default_rng(91)
        source = rng.normal(size=(160, 14))
        extra = rng.uniform(size=(160, 44))
        labels = (source[:, 0] > 0).astype(int)
        frame = tabular_input_frame(source, extra, 'X14_SIFE44', 'lightgbm')
        estimator = make_estimator('lightgbm', dict(n_estimators=300, lr=0.01, num_leaves=31), 12, 2)
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            estimator.fit(frame, labels)
            check_estimator_columns(estimator, frame)
            scores = estimator.predict_proba(frame, validate_features=True)[:, 1]
        self.assertTrue(np.isfinite(scores).all())
        self.assertFalse(any(('valid feature names' in str(w.message) for w in observed)))
        self.assertEqual(tuple(estimator.feature_name_), tuple(tabular_column_names('X14_SIFE44')))
        numeric = make_estimator('lightgbm', dict(n_estimators=300, lr=0.01, num_leaves=31), 12, 2)
        numeric.fit(frame.to_numpy(), labels)
        np.testing.assert_allclose(scores, numeric.booster_.predict(frame.to_numpy()), rtol=1e-12, atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'names/order'):
            check_estimator_columns(estimator, frame.iloc[:, ::-1])
if __name__ == '__main__':
    unittest.main()
