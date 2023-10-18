from __future__ import annotations

import pickle
import sys

import awkward as ak
import numpy as np
import tiled.client.awkward
import uproot
from uproot._util import no_filter


class TiledBranch:
    def __init__(self, name, interpretation):
        self._name = name
        self._interpretation = interpretation

    @property
    def name(self):
        return self._name

    @property
    def interpretation(self):
        return self._interpretation

    @property
    def typename(self):
        if self.interpretation is None:
            return "unknown"
        return self.interpretation.typename


class TiledUproot:
    def __init__(self, client: tiled.client.awkward.AwkwardClient):
        self._client = client
        self._name = None
        self._keys = None
        self._items = None
        self._offsets = ak.to_numpy(client[0, "offsets"])
        self._prefix = {}
        self._filenames = {}
        self._eras = {}
        self._treenames = {}
        self._interpretations = {}
        self._seekdata = {}

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
                (k, TiledBranch(k, pickle.loads(values[i]))) for k, i in where.items()
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
    def num_entries(self):
        return self._offsets[-1]

    def _file_index(self, entry):
        if 0 <= entry < self._offsets[-1]:
            return np.searchsorted(self._offsets, entry, side="right") - 1
        return None

    def _file_index_range(self, entry_start, entry_stop):
        e1 = max(0, min(self._offsets[-1], entry_start))
        e2 = max(e1, min(self._offsets[-1], entry_stop))
        if e1 == e2:
            return 0, 0

        i1, i2 = np.searchsorted(self._offsets, [e1, e2], side="right") - 1
        if self._offsets[i2] != e2:
            i2 += 1
        return i1, i2

    def _fetch_filedata(self, index_start, index_stop):
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
                self._eras[index_start + j] = era_index

    def _fetch_seekdata(self, index_start, index_stop, branches):
        branches_set = set(branches)
        if not all(
            len(branches_set.difference(self._seekdata.get(i, ()))) == 0
            for i in range(index_start, index_stop)
        ):
            fetched = self.client[0, "file", index_start:index_stop, "tree", branches]

            for j, data in enumerate(fetched):
                i = index_start + j

                seekdata = self._seekdata.get(i)
                if seekdata is None:
                    seekdata = self._seekdata[i] = {}

                for name in branches:
                    if name not in seekdata:
                        seekdata[name] = data[name]

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

        num = self.num_entries
        if entry_start is None:
            entry_start = 0
        elif entry_start < 0:
            entry_start += num
        entry_start = min(num, max(0, entry_start))

        if entry_stop is None:
            entry_stop = num
        elif entry_stop < 0:
            entry_stop += num
        entry_stop = min(num, max(0, entry_stop))

        if entry_stop < entry_start:
            entry_stop = entry_start

        index_start, index_stop = self._file_index_range(entry_start, entry_stop)

        # fetch from Tiled
        self.name  # noqa: B018
        self.keys()
        self._fetch_filedata(index_start, index_stop)

        return uproot.behaviors.TBranch.HasBranches.arrays(
            TiledUproot._FakeTree(
                self, entry_start, entry_stop, index_start, index_stop
            ),
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

    class _FakeBranch:
        def __init__(self, parent, name):
            self._parent = parent
            self._name = name
            self._interpretation = None

        @property
        def parent(self):
            return self._parent

        @property
        def name(self):
            return self._name

        @property
        def tree(self):
            return self._parent

        @property
        def interpretation(self):
            if self._interpretation is None:
                tree = self._parent
                tu = tree._parent
                for i in range(tree._index_start, tree._index_stop):
                    interp = tu._interpretations[tu._eras[i]]
                    if self._interpretation is None:
                        self._interpretation = interp
                    elif self._interpretation.cache_key != interp.cache_key:
                        msg = f"interpretation of TBranch {self._name!r} changed in file\n\n    {tu._filenames[i]}\n\nset entry_stop={tu._offsets[i]} to avoid it"
                        raise TypeError(msg)

                tree._branches_touched.append(self._name)

            return self._interpretation

        @property
        def cache_key(self):
            return self._name

        def entries_to_ranges_or_baskets(self, entry_start, entry_stop):
            self._parent.maybe_fetch_seekdata()

            out = []
            tree = self._parent
            tu = tree._parent
            for i in range(tree._index_start, tree._index_stop):
                global_start = tu._offsets[i]

                for seekdata in tu._seekdata[i][self._name]:
                    global_stop = tu._offsets[i] + seekdata["stop"]

                    if (
                        global_stop > global_start
                        and entry_start < global_stop
                        and global_start <= entry_stop
                    ):
                        byte_start = seekdata["seek"]
                        byte_stop = byte_start + seekdata["bytes"]
                        out.append((len(out), (byte_start, byte_stop)))
                        tree._range_to_file_index.append(i)

                    global_start = global_stop

            return out

    class _FakeTree:
        def __init__(self, parent, entry_start, entry_stop, index_start, index_stop):
            self._parent = parent
            self._entry_start = entry_start
            self._entry_stop = entry_stop
            self._index_start = index_start
            self._index_stop = index_stop

            self._file = TiledUproot._FakeFile(self)
            self._branches = [TiledUproot._FakeBranch(self, x) for x in self._parent]
            self._branchesdict = {x.name: x for x in self._branches}
            self._branches_touched = []
            self._fetched = False
            self._range_to_file_index = []

        @property
        def parent(self):
            return self._parent

        def itervalues(
            self,
            *,
            filter_name=no_filter,  # noqa: ARG002
            filter_typename=no_filter,  # noqa: ARG002
            filter_branch=no_filter,  # noqa: ARG002
            recursive=True,  # noqa: ARG002
        ):
            return self._branches

        @property
        def tree(self):
            return self

        @property
        def num_entries(self):
            return self._parent.num_entries

        @property
        def aliases(self):
            return {}

        def get(self, name):
            return self._branchesdict[name]

        def maybe_fetch_seekdata(self):
            if not self._fetched:
                self._parent._fetch_seekdata(
                    self._index_start, self._index_stop, self._branches_touched
                )
                self._fetched = True

    class _FakeFile:
        def __init__(self, parent):
            self._parent = parent
            self._source = TiledUproot._FakeSource(self)

    class _FakeSource:
        def __init__(self, parent):
            self._parent = parent
