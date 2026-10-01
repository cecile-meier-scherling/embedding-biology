"""Download and conservatively join Broad, Tox21 and FDA DILIrank annotations.

No compound names or private measurements are sent to an external service.
Bulk public downloads are cached with checksums; matching is entirely local.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import unicodedata

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/phh_prod_image_data_oasis_with_dmso.parquet'
OUT = ROOT / 'data/public_annotations'
SOURCES = {
    'broad_drugs_20250818.tsv': ('Broad Drug Repurposing Hub, 2025-08-18',
        'https://repo-hub.broadinstitute.org/public/data/repo-drug-annotation-20250818.txt'),
    'broad_samples_20250818.tsv': ('Broad Drug Repurposing Hub samples, 2025-08-18',
        'https://repo-hub.broadinstitute.org/public/data/repo-sample-annotation-20250818.txt'),
    'tox21_moleculenet.csv.gz': ('Tox21, MoleculeNet/DeepChem curated benchmark snapshot',
        'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'),
    'fda_dilirank_2.html': ('FDA DILIrank 2.0 official published table',
        'https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-20-dataset'),
}
TOX_TASKS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
BROAD_FIELDS = ['moa', 'target', 'clinical_phase', 'disease_area', 'indication']
# Only recognized small counterions may be removed from multi-organic structures.
COUNTERION_SMILES = [
    'CC(=O)O', 'O=C(O)C=CC(=O)O', 'O=C(O)/C=C/C(=O)O', 'O=C(O)/C=C\\C(=O)O',
    'O=C(O)C(O)C(O)C(=O)O', 'CS(=O)(=O)O', 'Cc1ccc(S(=O)(=O)O)cc1',
    'O=C(O)CC(O)(CC(=O)O)C(=O)O', 'O=C(O)C(=O)O', 'CC(O)C(=O)O',
    'O=S(=O)(O)c1ccccc1', 'O=C(O)CCC(=O)O',
]
UNCHARGER = rdMolStandardize.Uncharger()
COUNTERIONS = {Chem.MolToSmiles(UNCHARGER.uncharge(Chem.MolFromSmiles(s)), isomericSmiles=False)
               for s in COUNTERION_SMILES}


def normalized_name(value):
    """Normalize case/spacing only; preserve salts, prefixes, and stereochemistry."""
    if pd.isna(value):
        return ''
    return ' '.join(unicodedata.normalize('NFKC', str(value)).strip().casefold().split())


@lru_cache(maxsize=40000)
def structure_identity(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return ('', '', 'missing_structure')
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ('', '', 'invalid_structure')
        if any(g.GetGroupType() != Chem.StereoGroupType.STEREO_ABSOLUTE for g in mol.GetStereoGroups()):
            return ('', '', 'relative_or_mixed_stereochemistry')
        mol = rdMolStandardize.Cleanup(mol)
        fragments = Chem.GetMolFrags(mol, asMols=True)
        organic = [m for m in fragments if any(a.GetAtomicNum() == 6 for a in m.GetAtoms())]
        if len(organic) > 1:
            non_salts = [m for m in organic if Chem.MolToSmiles(UNCHARGER.uncharge(m), isomericSmiles=False) not in COUNTERIONS]
            if len(non_salts) != 1:
                return ('', '', 'multiple_organic_components')
            mol = non_salts[0]
        else:
            mol = rdMolStandardize.FragmentParent(mol)
        mol = UNCHARGER.uncharge(mol)
        parent = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        connectivity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        return parent, connectivity, 'ok'
    except (ValueError, RuntimeError):
        return ('', '', 'standardization_error')


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row = []
        if tag in ('th', 'td') and self.row is not None:
            self.cell = []

    def handle_data(self, text):
        if self.cell is not None:
            self.cell.append(text)

    def handle_endtag(self, tag):
        if tag in ('th', 'td') and self.cell is not None:
            self.row.append(''.join(self.cell).strip())
            self.cell = None
        if tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def parse_dilirank(path):
    parser = TableParser()
    parser.feed(path.read_text())
    rows = [r for r in parser.rows if len(r) == 6 and re.fullmatch(r'LT\d+', r[0])]
    result = pd.DataFrame(rows, columns=['ltkb_id', 'compound_name', 'severity_class', 'label_section', 'dili_concern', 'comment'])
    if len(result) != 1336 or result.ltkb_id.nunique() != 1336:
        raise ValueError(f'FDA table changed or incomplete: expected 1336 unique drugs, found {len(result)}')
    result['dili_concern'] = result.dili_concern.map(dili_category)
    expected = {'most': 217, 'less': 351, 'no': 414, 'ambiguous': 354}
    if result.dili_concern.value_counts().to_dict() != expected:
        raise ValueError('DILIrank category counts do not match the published 2.0 release')
    return result


def dili_category(value):
    text = re.sub(r'\s+', '', str(value)).casefold()
    for category in ['ambiguous', 'most', 'less', 'no']:
        if text in (f'{category}-dili-concern', f'v{category}-dili-concern'):
            return category
    raise ValueError(f'Unknown DILI category: {value}')


def consensus_binary(values):
    observed = {float(v) for v in values if pd.notna(v)}
    if not observed.issubset({0., 1.}):
        raise ValueError(f'Invalid binary labels: {observed}')
    if len(observed) == 1:
        return observed.pop(), 'observed_consensus'
    return np.nan, 'conflicting_measurements' if observed else 'unmeasured'


def acquire(raw, offline):
    manifest = []
    for filename, (source, url) in SOURCES.items():
        path = raw / filename
        if not path.exists():
            if offline:
                raise FileNotFoundError(path)
            temporary = path.with_suffix(path.suffix + '.partial')
            subprocess.run(['curl', '-fL', '--max-time', '120', '--retry', '2', url, '-o', str(temporary)], check=True)
            temporary.replace(path)
        manifest.append(dict(source=source, url=url, file=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                             bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             local_file_time_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()))
    return manifest


def local_compounds():
    names = sorted(pd.read_parquet(DATA, columns=['compound_id']).compound_id.dropna().astype(str).unique())
    cache = pd.read_csv(ROOT / 'results/chemical_split/pubchem_structure_lookup.csv', dtype=str).fillna('')
    if cache.compound_id.duplicated().any():
        raise ValueError('Duplicate compounds in existing PubChem cache')
    local = pd.DataFrame({'compound_id': names}).merge(cache, on='compound_id', how='left', validate='one_to_one').fillna('')
    overrides_path = DATA.parent / 'compound_structure_overrides.csv'
    if overrides_path.exists():
        overrides = pd.read_csv(overrides_path, dtype=str).fillna('').set_index('compound_id')
        if not overrides.index.is_unique:
            raise ValueError('Duplicate structure overrides')
        for i, row in local.iterrows():
            if row.compound_id in overrides.index and overrides.loc[row.compound_id, 'smiles']:
                local.loc[i, 'smiles'] = overrides.loc[row.compound_id, 'smiles']
                local.loc[i, 'status'] = 'manual_override'
    identities = [structure_identity(v) for v in local.smiles]
    local[['parent_smiles', 'connectivity_smiles', 'structure_status']] = pd.DataFrame(identities, index=local.index)
    local = local.rename(columns={'cid': 'cached_pubchem_cid', 'smiles': 'cached_smiles', 'status': 'cached_identity_status'})
    local['identity_group'] = [p if p else 'unresolved:' + n for p, n in zip(local.parent_smiles, local.compound_id)]
    local['name_key'] = local.compound_id.map(normalized_name)
    return local


def add_broad(local, drugs, samples):
    drugs = drugs.fillna('').copy()
    drugs['name_key'] = drugs.pert_iname.map(normalized_name)
    # Keep duplicate names: only transfer when every candidate annotation agrees.
    drugs = drugs.set_index('name_key', drop=False)
    samples = samples.fillna('').copy()
    samples['name_key'] = samples.pert_iname.map(normalized_name)
    samples['parent_smiles'] = [structure_identity(v)[0] for v in samples.smiles]
    by_parent, by_name = defaultdict(set), defaultdict(set)
    for row in samples.itertuples():
        if row.parent_smiles and row.name_key in drugs.index:
            by_parent[row.parent_smiles].add(row.name_key)
            by_name[row.name_key].add(row.parent_smiles)
    rows = []
    for row in local.itertuples():
        candidates, method = set(), 'unmatched'
        if row.name_key in drugs.index:
            if row.parent_smiles and by_name[row.name_key] and row.parent_smiles not in by_name[row.name_key]:
                method = 'name_structure_conflict'
            else:
                candidates, method = {row.name_key}, 'exact_name_and_parent' if row.parent_smiles and by_name[row.name_key] else 'exact_name_only'
        elif row.parent_smiles and row.parent_smiles in by_parent:
            candidates, method = by_parent[row.parent_smiles], 'exact_isomeric_parent'
        chosen = drugs.loc[sorted(candidates)] if candidates else drugs.iloc[:0]
        signatures = set(tuple(r) for r in chosen[BROAD_FIELDS].to_numpy())
        accepted = bool(candidates) and len(signatures) == 1
        if len(signatures) > 1:
            method = 'ambiguous_annotations_not_joined'
        result = dict(compound_id=row.compound_id, broad_match_method=method,
                      broad_candidates=';'.join(chosen.pert_iname), broad_matched=accepted)
        for field in BROAD_FIELDS:
            result['broad_' + field] = chosen.iloc[0][field] if accepted else ''
        rows.append(result)
    return pd.DataFrame(rows)


def add_tox21(local, tox):
    tox = tox.copy()
    tox['parent_smiles'] = [structure_identity(v)[0] for v in tox.smiles]
    tox['connectivity_smiles'] = [structure_identity(v)[1] for v in tox.smiles]
    valid = tox[tox.parent_smiles.ne('')]
    groups = {key: part for key, part in valid.groupby('parent_smiles')}
    connectivity = set(valid.connectivity_smiles)
    summaries, long_rows = [], []
    for row in local.itertuples():
        match = groups.get(row.parent_smiles)
        method = 'exact_isomeric_parent' if match is not None else (
            'connectivity_only_review_not_joined' if row.connectivity_smiles and row.connectivity_smiles in connectivity else 'unmatched')
        result = dict(compound_id=row.compound_id, tox21_match_method=method,
                      tox21_source_ids=';'.join(match.mol_id.astype(str)) if match is not None else '')
        for endpoint in TOX_TASKS:
            value, state = consensus_binary(match[endpoint]) if match is not None else (np.nan, 'unmatched')
            result['tox21__' + endpoint] = value
            if match is not None:
                long_rows.append(dict(compound_id=row.compound_id, source='Tox21_MoleculeNet', endpoint=endpoint,
                                      value=value, measurement_status=state, source_ids=result['tox21_source_ids'],
                                      match_method=method, input_identity_status=row.cached_identity_status))
        summaries.append(result)
    return pd.DataFrame(summaries), pd.DataFrame(long_rows), tox


def add_dili(local, broad_matches, dili):
    lookup = defaultdict(list)
    for row in dili.itertuples(index=False):
        lookup[normalized_name(row.compound_name)].append(row)
    broad = broad_matches.set_index('compound_id')
    result = []
    for row in local.itertuples():
        candidates = list(lookup.get(row.name_key, []))
        method = 'exact_name' if candidates else 'unmatched'
        b = broad.loc[row.compound_id]
        if not candidates and b.broad_matched:
            for alias in b.broad_candidates.split(';'):
                candidates.extend(lookup.get(normalized_name(alias), []))
            if candidates:
                method = 'via_broad_matched_name'
        categories = {c.dili_concern for c in candidates}
        category = next(iter(categories)) if len(categories) == 1 else ''
        if len(categories) > 1:
            method = 'conflicting_categories_not_joined'
        result.append(dict(compound_id=row.compound_id, dili_match_method=method,
                           dili_source_ids=';'.join(sorted({c.ltkb_id for c in candidates})),
                           dili_source_names=';'.join(sorted({c.compound_name for c in candidates})),
                           dili_category=category,
                           dili__most_vs_no={'most': 1., 'no': 0.}.get(category, np.nan),
                           dili__any_concern_vs_no={'most': 1., 'less': 1., 'no': 0.}.get(category, np.nan)))
    return pd.DataFrame(result)


def benchmark_support(joined):
    rows = []
    for column in [c for c in joined if c.startswith(('tox21__', 'dili__'))]:
        observed = joined[joined[column].notna()]
        # Count chemical identities rather than alias names as independent units.
        counts = observed.groupby('identity_group')[column].nunique()
        valid = observed[~observed.identity_group.isin(counts[counts > 1].index)]
        unique = valid.drop_duplicates('identity_group')
        positives, negatives = int(unique[column].eq(1).sum()), int(unique[column].eq(0).sum())
        rows.append(dict(endpoint=column, n_matched_names=len(observed), n_independent_identities=len(unique),
                         n_positive=positives, n_negative=negatives, n_identity_label_conflicts=int((counts > 1).sum()),
                         evaluation_ready=positives >= 20 and negatives >= 20,
                         recommendation='evaluate' if positives >= 20 and negatives >= 20 else 'inconclusive_low_class_support'))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Use cached files only; no network')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    raw = args.output / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    manifest = acquire(raw, args.offline)
    RDLogger.DisableLog('rdApp.warning')
    local = local_compounds()
    drugs = pd.read_csv(raw / 'broad_drugs_20250818.tsv', sep='\t', comment='!', dtype=str)
    samples = pd.read_csv(raw / 'broad_samples_20250818.tsv', sep='\t', comment='!', dtype=str)
    tox = pd.read_csv(raw / 'tox21_moleculenet.csv.gz')
    if len(tox) != 7831 or not set(TOX_TASKS).issubset(tox.columns):
        raise ValueError('Unexpected MoleculeNet Tox21 snapshot')
    dili = parse_dilirank(raw / 'fda_dilirank_2.html')
    print('Matching local compounds to Broad...', flush=True)
    broad = add_broad(local, drugs, samples)
    print('Matching Tox21 structures...', flush=True)
    tox_matches, tox_long, tox_normalized = add_tox21(local, tox)
    dili_matches = add_dili(local, broad, dili)
    joined = local.merge(broad, on='compound_id', validate='one_to_one').merge(tox_matches, on='compound_id', validate='one_to_one').merge(dili_matches, on='compound_id', validate='one_to_one')
    joined.to_parquet(args.output / 'compound_annotations.parquet', index=False)
    joined.to_csv(args.output / 'compound_annotations.csv', index=False)
    local.to_csv(args.output / 'local_identity_audit.csv', index=False)
    tox_long.to_csv(args.output / 'tox21_measurements.csv', index=False)
    tox_normalized.to_parquet(args.output / 'tox21_source_standardized.parquet', index=False)
    dili.to_csv(args.output / 'dilirank_2_source_table.csv', index=False)
    support = benchmark_support(joined)
    support.to_csv(args.output / 'endpoint_support.csv', index=False)
    audit_columns = ['compound_id', 'cached_identity_status', 'structure_status', 'broad_match_method', 'broad_candidates', 'tox21_match_method', 'dili_match_method']
    joined[audit_columns].to_csv(args.output / 'match_audit.csv', index=False)
    coverage = [
        dict(source='Broad Drug Repurposing Hub', source_records=len(drugs), matched_compounds=int(joined.broad_matched.sum())),
        dict(source='Tox21 MoleculeNet', source_records=len(tox), matched_compounds=int(joined.tox21_match_method.eq('exact_isomeric_parent').sum())),
        dict(source='FDA DILIrank 2.0', source_records=len(dili), matched_compounds=int(joined.dili_category.ne('').sum())),
    ]
    pd.DataFrame(coverage).assign(total_compounds=len(local)).to_csv(args.output / 'coverage.csv', index=False)
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(), sources=manifest, total_compounds=len(local),
                   matching='Exact case/spacing-normalized names or exact standardized isomeric parent structures. Connectivity-only matches excluded. No fuzzy matching.',
                   identity_caveat='Existing PubChem structures were obtained by name/first-hit lookup and are not independently verified. Their provenance is retained.',
                   tox21_missing='Unmeasured and contradictory duplicate outcomes are NaN, never negative.',
                   dili_missing='Ambiguous categories excluded from binary tasks; less-concern additionally excluded from most-vs-no.',
                   broad_negative_labels='Absent MOA/target entries are not experimentally established inactive controls.',
                   citations=['Corsello et al. Nature Medicine (2017), doi:10.1038/nm.4306', 'Wu et al. Chemical Science (2018), doi:10.1039/C7SC02664A', 'Olubamiwa et al. Drug Discovery Today (2025), 30(11):104485'],
                   broad_terms='Downloaded files retain their original non-commercial-use header; current portal describes CC-BY 4.0 metadata. Preserve source attribution and consult publisher terms for redistribution.')
    (args.output / 'manifest.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(pd.DataFrame(coverage).to_string(index=False), flush=True)
    print(support.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
