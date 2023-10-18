from __future__ import annotations

import pickle
import sys
from collections.abc import Mapping

import awkward as ak
import numpy as np
import tiled.client.awkward
import uproot
from uproot._util import no_filter


class _TiledBranch:
    def __init__(self, name, interpretation, parent):
        self._name = name
        self._interpretation = interpretation
        self._parent = parent

    @property
    def name(self):
        return self._name

    @property
    def interpretation(self):
        return self._interpretation

    @property
    def parent(self):
        return self._parent

    @property
    def typename(self):
        if self.interpretation is None:
            return "unknown"
        return self.interpretation.typename

    @property
    def cache_key(self):
        return self.name

    def entries_to_ranges_or_baskets(self, entry_start, entry_stop):
        # print(entry_start, entry_stop)
        # print(repr(self._parent.offsets))

        raise NotImplementedError


class _SeekData:
    def __init__(self, offsets, seeks, sizes):
        self._offsets = offsets
        self._seeks = seeks
        self._sizes = sizes

    @property
    def offsets(self):
        return self._offsets

    @property
    def seeks(self):
        return self._seeks

    @property
    def sizes(self):
        return self._sizes


class TiledUproot(Mapping):
    def __init__(self, client: tiled.client.awkward.AwkwardClient):
        self._client = client
        self._file = None
        self._name = None
        self._keys = None
        self._items = None
        self._offsets = None
        self._prefix = {}
        self._filenames = {}
        self._treenames = {}
        self._interpretations = {}
        self._seekdata = {}

    @property
    def tree(self):
        return self

    @property
    def client(self):
        return self._client

    @property
    def name(self):
        if self._name is None:
            self._name = self.client[0, "era", -1, "treename"]
        return self._name

    def keys(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,  # noqa: ARG002
        full_paths=True,  # noqa: ARG002
    ):
        if (
            filter_name is not no_filter
            or filter_typename is not no_filter
            or filter_branch is not no_filter
        ):
            raise NotImplementedError

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

    iterkeys = keys

    def items(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,  # noqa: ARG002
        full_paths=True,  # noqa: ARG002
    ):
        if (
            filter_name is not no_filter
            or filter_typename is not no_filter
            or filter_branch is not no_filter
        ):
            raise NotImplementedError

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
            self._items = [
                (k, _TiledBranch(k, pickle.loads(values[i]), self))
                for k, i in where.items()
            ]
        return self._items

    iteritems = items

    def values(
        self,
        *,
        filter_name=no_filter,  # noqa: ARG002
        filter_typename=no_filter,  # noqa: ARG002
        filter_branch=no_filter,  # noqa: ARG002
        recursive=True,  # noqa: ARG002
    ):
        return [v for k, v in self.items()]

    itervalues = values

    @property
    def branches(self):
        return self.values()

    def __iter__(self):
        yield from self.keys()

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, where):
        for k, v in self.items():
            if k == where:
                return v
        raise KeyError(where)

    def show(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
        full_paths=True,
        name_width=20,
        typename_width=24,
        interpretation_width=30,
        stream=sys.stdout,
    ):
        return uproot.behaviors.TBranch.HasBranches.show(
            self,
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=recursive,
            full_paths=full_paths,
            name_width=name_width,
            typename_width=typename_width,
            interpretation_width=interpretation_width,
            stream=stream,
        )

    @property
    def offsets(self):
        if self._offsets is None:
            self._offsets = ak.to_numpy(self.client[0, "offsets"])
        return self._offsets

    @property
    def num_entries(self):
        return self.offsets[-1]

    def file_index(self, entry):
        offs = self.offsets
        if 0 <= entry < offs[-1]:
            return np.searchsorted(offs, entry, side="right") - 1
        return None

    def file_index_range(self, entry_start, entry_stop):
        offs = self.offsets

        e1 = max(0, min(offs[-1], entry_start))
        e2 = max(e1, min(offs[-1], entry_stop))
        if e1 == e2:
            return 0, 0

        i1, i2 = np.searchsorted(offs, [e1, e2], side="right") - 1
        if offs[i2] != e2:
            i2 += 1
        return i1, i2

    def fetch_filedata(self, index_start, index_stop):
        if not all(i in self._filenames for i in range(index_start, index_stop)):
            fetched = self.client[
                0, "file", index_start:index_stop, ["filename", "era", "prefix"]
            ]

            for j, data in enumerate(fetched):
                era_index = data["era"]
                if era_index not in self._treenames:
                    era = self.client[0, "era", era_index]
                    self._treenames[era_index] = era["treename"]
                    self._interpretations[era_index] = dict(
                        zip(
                            era["names"],
                            [pickle.loads(x) for x in era["interpretations"]],
                        )
                    )

                prefix_index = data["prefix"]
                prefix = self._prefix.get(prefix_index)
                if prefix is None:
                    prefix = self._prefix[prefix_index] = self.client[
                        0, "prefix", prefix_index
                    ]

                self._filenames[index_start + j] = prefix + data["filename"]

    # def fetch_seekdata(self, index_start, index_stop, branches):
    #     if not all(i in self._seekdata for i in range(index_start, index_stop)):
    #         HERE

    #     for i in range(index_start, index_stop):
    #         if i not in self._seekdata:
    #             self._seekdata[i] = {}

    #         missing_branches = branches.difference(self._seekdata[i])
    #         fetched = self.client[0, "file", i, "tree", missing_branches]

    #         for name in missing_branches:
    #             offsets = np.empty(len(fetched[name]) + 1)
    #             offsets[0] = 0
    #             offsets[1:] = ak.to_numpy(fetched[name, "stop"])

    #             self._seekdata[i][name] = (
    #             )

    @property
    def aliases(self):
        return {}

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
        if decompression_executor is None:
            decompression_executor = uproot.source.futures.TrivialExecutor()
        if interpretation_executor is None:
            interpretation_executor = uproot.source.futures.TrivialExecutor()

        return uproot.behaviors.TBranch.HasBranches.arrays(
            self,
            expressions=expressions,
            cut=cut,
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            aliases=aliases,
            language=language,
            entry_start=entry_start,
            entry_stop=entry_stop,
            decompression_executor=decompression_executor,
            interpretation_executor=interpretation_executor,
            array_cache=array_cache,
            library=library,
            ak_add_doc=ak_add_doc,
            how=how,
        )
