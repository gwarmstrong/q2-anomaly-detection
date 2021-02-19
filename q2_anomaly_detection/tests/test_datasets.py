from unittest import TestCase
from q2_anomaly_detection.datasets import (
    Dataset,
    FileSystemArchiver,
    QiitaClient,
)


class AllExistArchiver:

    def __init__(self, *args):
        pass

    def exists(self, *args):
        return True

    def read(self, data):
        return data


class NoneExistArchiver:

    def __init__(self, *args):
        pass

    def exists(self, *args):
        return False


class AlternatingExistsArchiver:
    def __init__(self, *args):
        self.state = False

    def exists(self, *args):
        state = self.state
        self.state = not self.state
        return state


class DownloadCounter:

    def __init__(self, *args):
        self.download_count = 0

    def download(self, *args):
        self.download_count += 1


class DatasetTestCase(TestCase):

    def setUp(self) -> None:
        Dataset.archiver_type = NoneExistArchiver
        Dataset.client_type = DownloadCounter

    def tearDown(self) -> None:
        Dataset.archiver_type = FileSystemArchiver
        Dataset.client_type = QiitaClient

    def test_download(self):
        dataset = Dataset('some/path')

        dataset.download()
        self.assertEqual(dataset.client.download_count, 0)

        dataset.artifacts = {'foo': 'bar', 'baz': 'qux'}
        dataset.download()
        self.assertEqual(dataset.client.download_count, 2)
        dataset.artifacts['quux'] = 'corge'
        dataset.download()
        self.assertEqual(dataset.client.download_count, 5)
        dataset.archiver = AllExistArchiver()
        dataset.download()
        self.assertEqual(dataset.client.download_count, 5)

    def test_check_integrity(self):

        dataset = Dataset('some/path')
        self.assertTrue(dataset._check_integrity())

        dataset.artifacts = {'foo': 'bar', 'baz': 'qux'}
        self.assertFalse(dataset._check_integrity())

        dataset.archiver = AllExistArchiver()
        self.assertTrue(dataset._check_integrity())

        dataset.archiver = AlternatingExistsArchiver()
        # we expect this to fail because this mock object will return
        #  [False, True] because there are two artifacts and existence
        #  alternates
        self.assertFalse(dataset._check_integrity())

    def test_getitem(self):
        Dataset.archiver_type = AllExistArchiver
        dataset = Dataset('some/path')
        dataset.artifacts = {'foo': 'bar', 'baz': 'qux'}
        self.assertEqual(dataset['foo'], 'bar')
        self.assertEqual(dataset['baz'], 'qux')
        with self.assertRaises(KeyError):
            dataset['qux']
