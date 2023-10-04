from __future__ import annotations

import numbers
import pickle
import threading

import awkward as ak
import tiled.client
import uproot


class CollectedData:
    def __init__(self, prefix_depth=1):
        assert isinstance(prefix_depth, numbers.Integral) and prefix_depth >= 1
        self.prefix_depth = prefix_depth

        self.lock = threading.Lock()
        self.eras = {}
        self.entry_offsets = [0]
        self.prefix_eras = {}
        self.lookup_table = []

    def collect(self, filename_treename):
        filename, treename = filename_treename

        with uproot.open({filename: None}) as file:
            try:
                tree = file[treename]
            except:
                num_entries = 0
            else:
                branches = tree.values(recursive=True)

                era_key = f"{treename}\n" + "\n".join(
                    f"{branch.name}\t{branch.interpretation.cache_key}"
                    for branch in sorted(branches, key=lambda x: x.name)
                )

                num_entries = tree.num_entries

                branch_data = []
                for branch in branches:
                    num_baskets = branch._num_normal_baskets
                    branch_data.append(
                        [
                            {"seek": x, "stop": y, "bytes": z}
                            for x, y, z in zip(
                                branch.member("fBasketSeek")[:num_baskets],
                                branch.member("fBasketEntry")[1 : num_baskets + 1],
                                branch.member("fBasketBytes")[:num_baskets],
                            )
                        ]
                    )
                    if num_entries > 0 and num_baskets > 0:
                        assert branch_data[-1][-1]["stop"] == num_entries

        if num_entries > 0:
            with self.lock:
                era = self.eras.get(era_key)
                if era is None:
                    era = self.eras[era_key] = {
                        "index": len(self.eras),
                        "treename": treename,
                        "names": [branch.name for branch in branches],
                        "interpretations": [
                            pickle.dumps(branch.interpretation) for branch in branches
                        ],
                    }

                self.entry_offsets.append(self.entry_offsets[-1] + num_entries)

                split_filename = filename.split("/")
                assert len(split_filename) >= self.prefix_depth

                prefix = "/".join(split_filename[:-self.prefix_depth]) + "/"
                last_name = "/".join(split_filename[-self.prefix_depth:])
                prefix_era = self.prefix_eras.get(prefix)
                if prefix_era is None:
                    prefix_era = self.prefix_eras[prefix] = {
                        "index": len(self.prefix_eras),
                        "prefix": prefix,
                    }

                self.lookup_table.append(
                    {
                        "filename": last_name,
                        "tree": {
                            name: branch_data[i] for i, name in enumerate(era["names"])
                        },
                        "era": era["index"],
                        "prefix": prefix_era["index"],
                    }
                )

    def to_array(self):
        with self.lock:
            return ak.Array(
                [
                    {
                        "offsets": self.entry_offsets,
                        "file": self.lookup_table,
                        "era": [
                            {
                                "treename": x["treename"],
                                "names": x["names"],
                                "interpretations": x["interpretations"],
                            }
                            for x in self.eras.values()
                        ],
                        "prefix": [x["prefix"] for x in self.prefix_eras.values()],
                    }
                ]
            )

def concatenate(arrays):
    offsets = [0]
    file = []
    era = []
    num_eras = 0
    prefix = []
    num_prefixes = 0
    for array in arrays:
        offsets.extend(offsets[-1] + array[0, "offsets", 1:])
        array_file = array[0, "file"]
        array_file["era"] = num_eras + array_file["era"]
        array_file["prefix"] = num_prefixes + array_file["prefix"]
        file.append(array_file)
        era.append(array[0, "era"])
        num_eras += len(era[-1])
        prefix.append(array[0, "prefix"])
        num_prefixes += len(prefix[-1])

    return ak.Array(
        ak.contents.RecordArray(
            [
                ak.from_iter([offsets], highlevel=False),
                ak.contents.ListOffsetArray(
                    ak.index.Index64([0, sum(len(x) for x in file)]),
                    ak.concatenate(file, axis=0, highlevel=False),
                ),
                ak.contents.ListOffsetArray(
                    ak.index.Index64([0, sum(len(x) for x in era)]),
                    ak.concatenate(era, axis=0, highlevel=False),
                ),
                ak.contents.ListOffsetArray(
                    ak.index.Index64([0, sum(len(x) for x in prefix)]),
                    ak.concatenate(prefix, axis=0, highlevel=False),
                ),
            ],
            ["offsets", "file", "era", "prefix"],
        )
    )


def upload_to_tiled(url, name, lookup_table):
    client = tiled.client.from_uri(url)
    client.write_awkward(lookup_table, key=name)


if __name__ == "__main__":
    files = [
        {
            "/home/jpivarski/storage/data/Run2018D-DoubleMuon-Nano25Oct2019_ver2-v1-974F28EE-0FCE-4940-92B5-870859F880B1.root": "Events"
        }
    ]
    files = uproot._util.regularize_files(files, steps_allowed=False)

    collected_data = CollectedData()
    for filename_treename in files:
        collected_data.collect(filename_treename)

    array = collected_data.to_array()

    final = concatenate([array, array, array])

    # upload_to_tiled(
    #     "http://127.0.0.1:8000?api_key=5b0076d3e33202d55884b2428a4f405e3ca84510c8c6b60fbe60d393998abbb0",
    #     "dataset_name",
    #     lookup_table,
    # )
