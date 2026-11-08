
import unittest
from src.pubmed_client import parse_diseases
from unittest.mock import patch, Mock
from src.pubmed_client import search_pmids


class TestPubMedClient(unittest.TestCase):

    def test_single_disease(self):
        result = parse_diseases("Ovarian Cancer")
        self.assertEqual(result, ["Ovarian Cancer"])

    def test_multiple_diseases(self):
        result = parse_diseases(
            "['Aging', 'Inflammation', 'Frailty']"
        )
        self.assertEqual(
            result,
            ["Aging", "Inflammation", "Frailty"]
        )

    def test_missing_disease(self):
        result = parse_diseases(None)
        self.assertEqual(result, [])

    
    @patch("src.pubmed_client.resolve_mesh_term", return_value=None)
    @patch("src.pubmed_client.resolve_pubchem_name", return_value=None)
    @patch("src.pubmed_client.time.sleep")
    @patch("src.pubmed_client._get")
    def test_two_stage_search(self, mock_get, mock_sleep, mock_pubchem, mock_mesh):
        specific = Mock()
        specific.json.return_value = {
            "esearchresult": {"idlist": ["101", "102"]}
        }

        general = Mock()
        general.json.return_value = {
            "esearchresult": {"idlist": ["102", "103"]}
        }

        mock_get.side_effect = [specific, general]

        result = search_pmids(
            "Topotecan",
            "2020-01-01",
            "Ovarian Cancer"
        )

        self.assertEqual(result, ["101", "102", "103"])
        self.assertEqual(mock_get.call_count, 2)

        specific_query = mock_get.call_args_list[0].args[1]["term"]
        self.assertIn("Topotecan", specific_query)
        self.assertIn("Ovarian Cancer", specific_query)


    @patch("src.pubmed_client.resolve_mesh_term", return_value=None)
    @patch("src.pubmed_client.resolve_pubchem_name", return_value=None)
    @patch("src.pubmed_client.time.sleep")
    @patch("src.pubmed_client._get")
    def test_fallback_to_drug_only(self, mock_get, mock_sleep, mock_pubchem, mock_mesh):
        specific = Mock()
        specific.json.return_value = {
            "esearchresult": {"idlist": []}
        }

        general = Mock()
        general.json.return_value = {
            "esearchresult": {"idlist": ["201", "202"]}
        }

        mock_get.side_effect = [specific, general]

        result = search_pmids(
            "Basmisanil",
            "2020-01-01",
            "Stroke"
        )

        self.assertEqual(result, ["201", "202"])
        self.assertEqual(mock_get.call_count, 2)




if __name__ == "__main__":
    unittest.main()
