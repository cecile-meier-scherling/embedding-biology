"""Identity and missing-label safeguards for public annotation ingestion."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
from import_public_annotations import (
    BROAD_FIELDS, TableParser, add_broad, add_dili, add_tox21,
    benchmark_support, consensus_binary, dili_category, normalized_name,
    structure_identity, TOX_TASKS,
)


class PublicAnnotationTests(unittest.TestCase):
    def test_names_preserve_stereo_and_salt(self):
        self.assertEqual(normalized_name('  ASPIRIN '), 'aspirin')
        self.assertNotEqual(normalized_name('(R)-drug'), normalized_name('(S)-drug'))
        self.assertNotEqual(normalized_name('drug hydrochloride'), normalized_name('drug'))

    def test_salts_and_enantiomers(self):
        self.assertEqual(structure_identity('CC[NH3+].[Cl-]')[0], structure_identity('CCN')[0])
        self.assertNotEqual(structure_identity('C[C@H](O)C(=O)O')[0], structure_identity('C[C@@H](O)C(=O)O')[0])
        self.assertEqual(structure_identity('CCN.CCCCO')[2], 'multiple_organic_components')

    def test_binary_missing_conflicts(self):
        self.assertEqual(consensus_binary([0., np.nan, 0.]), (0., 'observed_consensus'))
        self.assertEqual(consensus_binary([1., np.nan])[0], 1.)
        value, state = consensus_binary([0., 1.])
        self.assertTrue(np.isnan(value)); self.assertEqual(state, 'conflicting_measurements')
        self.assertTrue(np.isnan(consensus_binary([np.nan])[0]))
        with self.assertRaises(ValueError):
            consensus_binary([2.])

    def test_name_structure_conflict_is_not_joined(self):
        local = pd.DataFrame([dict(compound_id='drug', name_key='drug', parent_smiles=structure_identity('CCO')[0])])
        drugs = pd.DataFrame([dict(pert_iname='drug', **{f: 'annotation' for f in BROAD_FIELDS})])
        samples = pd.DataFrame([dict(pert_iname='drug', smiles='CCN')])
        result = add_broad(local, drugs, samples).iloc[0]
        self.assertFalse(result.broad_matched)
        self.assertEqual(result.broad_match_method, 'name_structure_conflict')
        self.assertEqual(result.broad_target, '')

    def test_duplicate_conflicting_annotations_are_not_joined(self):
        local = pd.DataFrame([dict(compound_id='drug', name_key='drug', parent_smiles='CCO')])
        drugs = pd.DataFrame([dict(pert_iname='drug', **{f: 'a' for f in BROAD_FIELDS}),
                              dict(pert_iname='DRUG', **{f: 'b' for f in BROAD_FIELDS})])
        samples = pd.DataFrame([dict(pert_iname='drug', smiles='CCO')])
        result = add_broad(local, drugs, samples).iloc[0]
        self.assertFalse(result.broad_matched)
        self.assertEqual(result.broad_match_method, 'ambiguous_annotations_not_joined')

    def test_connectivity_only_is_review_not_automatic(self):
        a, connection, _ = structure_identity('C[C@H](O)C(=O)O')
        local = pd.DataFrame([dict(compound_id='drug', parent_smiles=a, connectivity_smiles=connection,
                                   cached_identity_status='pubchem_first_match')])
        tox = pd.DataFrame([dict(smiles='C[C@@H](O)C(=O)O', mol_id='x', **{c: 1. for c in TOX_TASKS})])
        matches, _, _ = add_tox21(local, tox)
        self.assertEqual(matches.tox21_match_method.iloc[0], 'connectivity_only_review_not_joined')
        self.assertTrue(matches['tox21__NR-AR'].isna().all())

    def test_dili_ambiguous_is_not_negative(self):
        local = pd.DataFrame([dict(compound_id='drug', name_key='drug')])
        broad = pd.DataFrame([dict(compound_id='drug', broad_matched=False, broad_candidates='')])
        dili = pd.DataFrame([dict(compound_name='drug', ltkb_id='LT001', dili_concern='ambiguous')])
        joined = add_dili(local, broad, dili)
        self.assertTrue(joined['dili__any_concern_vs_no'].isna().all())
        self.assertEqual(dili_category('vMOST-DILI-concern'), 'most')
        with self.assertRaises(ValueError):
            dili_category('unknown')

    def test_nested_html_and_entities(self):
        parser = TableParser()
        parser.feed('<table><tr><td>LT001</td><td><b>A &amp; B</b></td><td><sup>v</sup>Most-DILI-concern</td></tr></table>')
        self.assertEqual(parser.rows, [['LT001', 'A & B', 'vMost-DILI-concern']])

    def test_support_counts_identities_not_aliases(self):
        data = pd.DataFrame({'compound_id':['a','a-salt','b','c'], 'identity_group':['A','A','B','C'],
                             'tox21__x':[1.,1.,0.,np.nan]})
        row = benchmark_support(data).iloc[0]
        self.assertEqual(row.n_independent_identities, 2)
        self.assertEqual(row.n_positive, 1)
        data.loc[1, 'tox21__x'] = 0.
        row = benchmark_support(data).iloc[0]
        self.assertEqual(row.n_identity_label_conflicts, 1)
        self.assertEqual(row.n_independent_identities, 1)


if __name__ == '__main__':
    unittest.main()
