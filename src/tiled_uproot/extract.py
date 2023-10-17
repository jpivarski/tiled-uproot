from __future__ import annotations

import pickle

import tiled.client.awkward
import uproot
from uproot._util import no_filter


class _TiledSource(uproot.source.chunk.Source):
    def __init__(self, client: tiled.client.awkward.AwkwardClient):
        self._client = client

    @property
    def client(self):
        return self._client


class _TiledFile:
    def __init__(self, client: tiled.client.awkward.AwkwardClient):
        self._source = _TiledSource(client)

    @property
    def source(self):
        return self._source

    @property
    def client(self):
        return self._source.client


class TiledUproot:
    def __init__(self, client: tiled.client.awkward.AwkwardClient):
        self._file = _TiledFile(client)
        self._name = None
        self._keys = None
        self._items = None
        self._num_entries = None

    @property
    def tree(self):
        return self

    @property
    def client(self):
        return self._file.client

    @property
    def name(self):
        if self._name is None:
            self._name = self.client[0, "era", -1, "treename"]
        return self._name

    def keys(self):
        if self._keys is None:
            if self._items is not None:
                self._keys = [k for k, v in self._items]
            else:
                seen = set()
                self._keys = []
                for era in self.client[0, "era", "names"]:
                    for name in era:
                        if name not in seen:
                            seen.add(name)
                            self._keys.append(name)
        return self._keys

    def items(self):
        if self._items is None:
            where = {}
            values = []
            for era in self.client[0, "era", ["names", "interpretations"]]:
                for name, interpretation in zip(era["names"], era["interpretations"]):
                    if name not in where:
                        where[name] = len(values)
                        values.append(interpretation)
                    else:
                        values[where[name]] = interpretation
            self._items = [(k, pickle.loads(values[i])) for k, i in where.items()]
        return self._items

    @property
    def num_entries(self):
        if self._num_entries is None:
            self._num_entries = self.client[0, "offsets", -1]
        return self._num_entries

    def arrays(
        self,
        expressions=None,
        cut=None,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        aliases=None,
        language=uproot.language.python.python_language,
        entry_start=None,
        entry_stop=None,
        decompression_executor=None,
        interpretation_executor=None,
        array_cache=None,
        library="ak",
        ak_add_doc=False,
        how=None,
    ):
        raise NotImplementedError
