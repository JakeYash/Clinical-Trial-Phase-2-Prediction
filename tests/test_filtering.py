import unittest
from unittest.mock import patch, Mock
import pandas as pd
from src.pubmed_client import (_date_window, organization_matches,
                               trial_organizations, fetch_abstracts, collect_drug_discussion)
from src.filters import run_all_filters
from src.merge import build_pubmed_summary_features, merge_trials_with_pubmed
from src.ctgov_client import flatten_study


class FilteringTests(unittest.TestCase):
    def test_dates(self):
        for value, expected in [('2020-03-01', '2020/02/29'),
                                ('2020-03', '2020/02/29'), ('2020', '2019/12/31')]:
            self.assertEqual(_date_window(value)[1], expected)
        for value in [None, float('nan'), '', 'bad', '2020-02-30']:
            self.assertIsNone(_date_window(value))

    def test_disclosed_ties(self):
        for field in ['affiliations', 'funding_agencies']:
            self.assertTrue(organization_matches({field: ['ACME, research division']}, ['Acme Inc.']))
        self.assertTrue(organization_matches({'conflict_of_interest': 'Consultant for Acme.'}, ['Acme']))
        self.assertFalse(organization_matches({'affiliations': ['Merckx University']}, ['Merck']))
        self.assertFalse(organization_matches({'title': 'Acme trial', 'abstract': 'Acme drug'}, ['Acme']))

    def test_xml_and_collection(self):
        xml = b'''<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID>
        <Article><ArticleTitle>Study</ArticleTitle><AuthorList><Author><AffiliationInfo>
        <Affiliation>Acme <b>Research</b></Affiliation></AffiliationInfo></Author></AuthorList>
        <GrantList><Grant><Agency>Acme</Agency></Grant></GrantList></Article>
        <CoiStatement>Paid by Acme</CoiStatement></MedlineCitation></PubmedArticle></PubmedArticleSet>'''
        with patch('src.pubmed_client._get', return_value=Mock(content=xml)), patch('src.pubmed_client.time.sleep'):
            records = fetch_abstracts(['1'])
        self.assertEqual(records[0]['affiliations'], ['Acme Research'])
        self.assertEqual(records[0]['funding_agencies'], ['Acme'])
        excluded = []
        with patch('src.pubmed_client.search_pmids', return_value=['1', '2']), patch('src.pubmed_client.fetch_abstracts', return_value=records + [{'pmid':'2'}]):
            kept = collect_drug_discussion('drug', '2020', organizations=['Acme'], excluded_records=excluded)
        self.assertEqual([r['pmid'] for r in kept], ['2'])
        # 3 matches because Acme appears in affiliations, funding_agencies, and conflict_of_interest
        self.assertEqual(len(excluded[0]['exclusion_matches']), 3)

    def test_missing_dates_retained_with_unknown_counts(self):
        trials = pd.DataFrame([dict(nct_id='A', label=1, start_date=None, condition='X', intervention_name='Y'),
                               dict(nct_id='B', label=0, start_date='2020', condition='X', intervention_name='Y')])
        trials = run_all_filters(trials)
        self.assertEqual(len(trials), 2)
        result = merge_trials_with_pubmed(trials, build_pubmed_summary_features({'A': None, 'B': []}))
        self.assertTrue(pd.isna(result.iloc[0].pubmed_mention_count))
        self.assertEqual(result.iloc[1].pubmed_mention_count, 0)
        self.assertEqual(result.iloc[0].pubmed_collection_status, 'missing_or_invalid_start_date')
        empty = merge_trials_with_pubmed(trials, build_pubmed_summary_features({}))
        self.assertTrue(empty.pubmed_mention_count.isna().all())

    def test_industry_collaborators_and_aliases(self):
        row = flatten_study({'protocolSection': {'sponsorCollaboratorsModule': {
            'leadSponsor': {'name': 'University', 'class': 'OTHER'},
            'collaborators': [{'name': 'Acme', 'class': 'INDUSTRY'}, {'name': 'Hospital', 'class': 'OTHER'}]}}})
        with patch('config.DRUG_MANUFACTURERS', {'drug': ['Maker', 'Former Maker']}):
            self.assertEqual(trial_organizations(row, 'Drug'), ['Acme', 'Former Maker', 'Maker'])

    def test_pipeline_skips_search_for_missing_date(self):
        import tempfile
        from pathlib import Path
        import pipeline
        with tempfile.TemporaryDirectory() as directory, patch('config.RAW_PUBMED_PATH', Path(directory)/'kept'), patch('config.RAW_PUBMED_EXCLUDED_PATH', Path(directory)/'excluded'), patch('pipeline.collect_drug_discussion') as collect:
            result = pipeline.step4_collect_pubmed(pd.DataFrame([{'nct_id': 'A', 'drugs': "['drug']", 'start_date': None}]))
        self.assertEqual(result, {'A': None})
        collect.assert_not_called()

if __name__ == '__main__':
    unittest.main()
