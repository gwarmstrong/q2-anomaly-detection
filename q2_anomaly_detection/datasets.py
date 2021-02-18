from abc import ABC
from urllib import request
from zipfile import ZipFile
import os
import pandas as pd
from biom import load_table
from q2_anomaly_detection.exceptions import DatasetError


def download_and_extract_response(response, path):
    os.makedirs(path, exist_ok=True)
    zip_path = path + '.zip'
    with open(zip_path, 'wb') as fp:
        fp.write(response.read())

    with ZipFile(zip_path, 'r') as fp:
        fp.extractall(path)
    os.remove(zip_path)


class ArchiverMixin:
    dataset: 'Dataset'
    table_kw: str
    metadata_kw: str


class QiitaSaveMixin(ArchiverMixin):

    def save_qiita_table(self, response):
        self._download_and_extract(response, self.table_kw)

    def save_qiita_metadata(self, response):
        self._download_and_extract(response, self.metadata_kw)

    def _download_and_extract(self, response, keyword):
        metadata_path = os.path.join(self.dataset.path,
                                     keyword
                                     )
        download_and_extract_response(response, metadata_path)


class _FilesystemArchiver(QiitaSaveMixin):
    table_kw = 'table'
    metadata_kw = 'metadata'

    def __init__(self, dataset):
        self.dataset: Dataset = dataset

    @property
    def metadata_path(self):
        path_to_metadata = os.path.join(
            self.dataset.path, self.metadata_kw, 'templates'
        )
        if not os.path.exists(path_to_metadata):
            return False
        md_candidates = list(
            filter(lambda n: n.endswith('.txt'),
                   os.listdir(path_to_metadata)
                   )
        )
        if len(md_candidates) == 0:
            return False
        else:
            md_file = md_candidates[0]
        full_path = os.path.join(path_to_metadata, md_file)
        return os.path.abspath(full_path)

    @property
    def read_metadata(self):
        return pd.read_csv(self.metadata_path, sep='\t')

    def read_table(self):
        return load_table(self.table_path)

    @property
    def table_path(self):
        path_to_table = os.path.join(
            self.dataset.path, self.table_kw, 'BIOM',
            str(self.dataset.table_artifact_id),
            'otu_table.biom',
        )
        return os.path.abspath(path_to_table)

    def metadata_exists(self):
        return True if self.metadata_path else False

    def table_exists(self):
        return os.path.exists(self.table_path)


class Artifact:
    pass


class QiitaArtifact:
    _artifact_fstring = "https://qiita.ucsd.edu/public_artifact_download/" \
                        "?artifact_id={}"
    _metadata_fstring = "https://qiita.ucsd.edu/public_download/" \
                        "?data=sample_information&study_id={}"


class Table(Artifact, QiitaArtifact):

    def __init__(self, name, artifact_id):
        self.name = name
        self.artifact_id = artifact_id

    def qiita_download(self, client: '_QiitaClient'):
        table_link = self._artifact_fstring.format(self.artifact_id)
        r = client.make_request(table_link)
        return r


class Metadata(Artifact, QiitaArtifact):

    def __init__(self, name, study_id):
        self.name = name
        self.study_id = study_id

    def qiita_download(self, client: '_QiitaClient'):
        metadata_link = self._metadata_fstring.format(
            self.study_id)
        r = client.make_request(metadata_link)
        return r


class _QiitaClient:

    _artifact_fstring = "https://qiita.ucsd.edu/public_artifact_download/" \
                        "?artifact_id={}"
    _metadata_fstring = "https://qiita.ucsd.edu/public_download/" \
                        "?data=sample_information&study_id={}"

    def __init__(self, dataset, archiver):
        self.dataset: Dataset = dataset
        self.archiver = archiver

    def download_table(self):
        table_link = self._artifact_fstring.format(
            self.dataset.table_artifact_id)
        r = self.make_request(table_link)
        self.archiver.save_qiita_table(r)

    def download_metadata(self):
        metadata_link = self._metadata_fstring.format(self.dataset.study_id)
        r = self.make_request(metadata_link)
        self.archiver.save_qiita_metadata(r)

    @staticmethod
    def make_request(url):
        return request.urlopen(url)


class Dataset(ABC):

    study_id: int
    table_artifact_id: int

    def __init__(self, path, download=True):
        self.path = path
        self.archiver = _FilesystemArchiver(self)
        self.client = _QiitaClient(self, self.archiver)

        if download:
            self.download()

        if not self._check_integrity():
            raise DatasetError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self._table = None
        self._metadata = None

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        self.client.download_metadata()
        self.client.download_table()

    def _check_integrity(self):
        return all([
                self.archiver.metadata_exists(),
                self.archiver.table_exists(),
            ])

    @property
    def table(self):
        if self._table is None:
            self._table = self.archiver.read_table()
        return self._table

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self.archiver.read_metadata()
        return self._metadata


# class MultiPrepDataset(Dataset):
#
#     table_artifact_id: List[int]
#
#     def __init__(self, path, download_ok=True):
#         super().__init__(path, download_ok=download_ok)
#         self.client.download_table = self._download_table
#
#     def _table_apply(self, fn):
#         return [fn(id_) for id_ in self.table_artifact_id]
#
#     def read_table(self):
#         tables = [load_table(path) for path in self.table_path]
#         t1 = tables.pop()
#         for t in tables:
#             t1 = t1.merge(t)
#         return t1
#
#     def _table_exists(self):
#         return all(self._single_table_exists(path) for path in self.table_path)
#
#     @property
#     def table_path(self):
#         return self._table_apply(self.client.single_table_path)
#
#     def _download_table(self):
#         self._table_apply(self.client.download_single_table)


class KeyboardDataset(Dataset):
    study_id = 232
    table_artifact_id = 46809


# class DietInterventionStudy(MultiPrepDataset):
#     study_id = 11550
#     table_artifact_id = [63512, 63515]
