from abc import ABC
from urllib import request
from zipfile import ZipFile
import os
import pandas as pd
from biom import load_table
from q2_anomaly_detection.exceptions import DatasetError


class Dataset(ABC):

    study_id: int
    table_artifact_id: int

    _artifact_fstring = "https://qiita.ucsd.edu/public_artifact_download/" \
                        "?artifact_id={}"
    _metadata_fstring = "https://qiita.ucsd.edu/public_download/" \
                        "?data=sample_information&study_id={}"

    _table_kw = 'table'
    _metadata_kw = 'metadata'

    def __init__(self, path, download_ok=True):
        self.path = path
        self._try_download('metadata', self._download_metadata, download_ok,
                           self._metadata_exists,
                           )
        self._try_download('table', self._download_table, download_ok,
                           self._table_exists,
                           )
        self._table = None
        self._metadata = None

    @staticmethod
    def _try_download(data_type, download_method, download_ok,
                      exists_method):
        exists = exists_method()
        if (not exists) and download_ok:
            download_method()
        elif not exists:
            raise DatasetError(f'Dataset {data_type} does not exist but '
                               f'download not allowed. Set '
                               f'`download_ok=True` to download data.')

    @staticmethod
    def _download_and_unzip(url, path):
        os.makedirs(path, exist_ok=True)
        r = request.urlopen(url)
        zip_path = path + '.zip'
        with open(zip_path, 'wb') as fp:
            fp.write(r.read())

        with ZipFile(zip_path, 'r') as fp:
            fp.extractall(path)
        os.remove(zip_path)

    def _download_table(self):
        table_link = self._artifact_fstring.format(self.table_artifact_id)
        table_path = os.path.join(self.path, self._table_kw)
        self._download_and_unzip(table_link, table_path)

    def _download_metadata(self):
        metadata_link = self._metadata_fstring.format(self.study_id)
        metadata_path = os.path.join(self.path, self._metadata_kw)
        self._download_and_unzip(metadata_link, metadata_path)

    @property
    def table_path(self):
        path_to_table = os.path.join(
            self.path, self._table_kw, 'BIOM', str(self.table_artifact_id),
            'otu_table.biom',
        )
        return os.path.abspath(path_to_table)

    @property
    def metadata_path(self):
        path_to_metadata = os.path.join(
            self.path, self._metadata_kw, 'templates'
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

    def _table_exists(self):
        return os.path.exists(self.table_path)

    def _metadata_exists(self):
        return True if self.metadata_path else False

    @property
    def table(self):
        if self._table is None:
            self._table = load_table(self.table_path)
        return self._table

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = pd.read_csv(self.metadata_path, sep='\t')
        return self._metadata


class KeyboardDataset(Dataset):
    study_id = 232
    table_artifact_id = 46809
