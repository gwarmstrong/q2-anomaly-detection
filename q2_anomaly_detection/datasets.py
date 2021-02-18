from typing import Dict, Callable
from abc import ABC
from urllib import request
from zipfile import ZipFile
import os
import pandas as pd
from biom import load_table
from q2_anomaly_detection.exceptions import DatasetError
from http.client import HTTPResponse


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
    save: Callable


class QiitaSaveMixin(ArchiverMixin):

    _download_method = download_and_extract_response

    def save(self, artifact: 'Artifact', response: HTTPResponse):
        path = os.path.join(self.dataset.path, artifact.name)
        self._download_method(response, path)


class FileSystemArchiver(QiitaSaveMixin):

    def __init__(self, dataset):
        self.dataset: Dataset = dataset
        self.path = self.dataset.path

    def read(self, artifact: 'FileSystemArtifact'):
        return artifact.filesystem_read(self)

    def exists(self, artifact: 'FileSystemArtifact'):
        return artifact.filesystem_exists(self)

    def path(self, artifact: 'FileSystemArtifact'):
        return artifact.filesystem_path(self)


class Artifact:
    name: str

    @staticmethod
    def merge(self, other):
        raise NotImplementedError()


class QiitaArtifact(Artifact):
    _artifact_fstring = "https://qiita.ucsd.edu/public_artifact_download/" \
                        "?artifact_id={}"
    _metadata_fstring = "https://qiita.ucsd.edu/public_download/" \
                        "?data=sample_information&study_id={}"

    def qiita_download(self, client: 'QiitaClient', archiver: ArchiverMixin):
        raise NotImplementedError('You should implement this.')


class FileSystemArtifact(Artifact):
    def filesystem_exists(self, archiver: FileSystemArchiver):
        raise NotImplementedError('You should implement this.')

    def filesystem_read(self, archiver: FileSystemArchiver):
        raise NotImplementedError('You should implement this.')

    def filesystem_path(self, archiver: FileSystemArchiver):
        raise NotImplementedError('You should implement this.')


class Table(QiitaArtifact, FileSystemArtifact):

    name = 'table'

    def __init__(self, artifact_id):
        self.artifact_id = artifact_id

    def qiita_download(self, client: 'QiitaClient', archiver: ArchiverMixin):
        table_link = self._artifact_fstring.format(self.artifact_id)
        r = client.make_request(table_link)
        archiver.save(self, r)

    def filesystem_exists(self, archiver: FileSystemArchiver):
        path = self.filesystem_path(archiver)
        return os.path.exists(path)

    def filesystem_read(self, archiver: FileSystemArchiver):
        path = self.filesystem_path(archiver)
        return load_table(path)

    def filesystem_path(self, archiver: FileSystemArchiver):
        path_to_table = os.path.join(
            archiver.path, self.name, 'BIOM',
            str(self.artifact_id),
            'otu_table.biom',
        )
        return os.path.abspath(path_to_table)

    @staticmethod
    def merge(t1, t2):
        return t1.merge(t2)


class ArtifactList(QiitaArtifact, FileSystemArtifact):

    def __init__(self, *artifacts):
        self.artifacts = list(artifacts)

    def qiita_download(self, client: 'QiitaClient', archiver: ArchiverMixin):
        for artifact in self.artifacts:
            artifact.qiita_download(client, archiver)

    def filesystem_exists(self, archiver: FileSystemArchiver):
        return all(
            artifact.filesystem_exists(archiver) for artifact in self.artifacts
        )

    def filesystem_read(self, archiver: FileSystemArchiver):
        if len(self.artifacts) > 0:
            artifact = self.artifacts[0]
            merged = artifact.filesystem_read(archiver)
        for artifact in self.artifacts[1:]:
            other = artifact.filesystem_read(archiver)
            merged = artifact.merge(merged, other)
        return merged


class Tables(ArtifactList):

    name = 'table'


class Metadata(QiitaArtifact, FileSystemArtifact):

    name = 'metadata'

    def __init__(self, study_id):
        self.study_id = study_id

    def qiita_download(self, client: 'QiitaClient', archiver: ArchiverMixin):
        metadata_link = self._metadata_fstring.format(
            self.study_id)
        r = client.make_request(metadata_link)
        archiver.save(self, r)

    def filesystem_exists(self, archiver):
        return True if self.filesystem_path(archiver) else False

    def filesystem_read(self, archiver):
        path = self.filesystem_path(archiver)
        return pd.read_csv(path, sep='\t')

    def filesystem_path(self, archiver):
        path_to_metadata = os.path.join(
            archiver.path, self.name, 'templates'
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


class QiitaClient:

    _artifact_fstring = "https://qiita.ucsd.edu/public_artifact_download/" \
                        "?artifact_id={}"
    _metadata_fstring = "https://qiita.ucsd.edu/public_download/" \
                        "?data=sample_information&study_id={}"

    def __init__(self, dataset, archiver: QiitaSaveMixin):
        self.dataset: Dataset = dataset
        self.archiver = archiver

    def download(self, artifact: QiitaArtifact):
        artifact.qiita_download(self, self.archiver)

    @staticmethod
    def make_request(url):
        return request.urlopen(url)


class Dataset(ABC):

    artifacts: Dict[str, Artifact] = dict()
    archiver_type = FileSystemArchiver
    client_type = QiitaClient

    def __init__(self, path, download=True):
        self.path = path
        self.archiver = self.archiver_type(self)
        self.client = self.client_type(self, self.archiver)

        if download:
            self.download()

        if not self._check_integrity():
            raise DatasetError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self._table = None
        self._metadata = None
        self._data = dict()

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        for artifact in self.artifacts.values():
            self.client.download(artifact)

    def _check_integrity(self):
        return all(
                self.archiver.exists(artifact) for artifact in
                self.artifacts.values()
            )

    def __getitem__(self, item):
        if item in self._data:
            return self._data[item]
        elif item in self.artifacts:
            value = self.archiver.read(self.artifacts[item])
            self._data[item] = value
            return value
        else:
            raise KeyError(item)


class KeyboardDataset(Dataset):
    study_id = 232
    table_artifact_id = 46809

    artifacts = {
        'metadata': Metadata(study_id),
        'table': Table(table_artifact_id),
    }


class DietInterventionStudy(Dataset):
    study_id = 11550
    table_artifact_ids = [63512, 63515]

    artifacts = {
        'metadata': Metadata(study_id),
        'table': Tables(
            Table(table_artifact_ids[0]),
            Table(table_artifact_ids[1]),
        ),
    }
