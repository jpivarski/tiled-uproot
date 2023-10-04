from __future__ import annotations

import pickle
import threading

import awkward as ak
import tiled.client
import uproot


class CollectedData:
    def __init__(self):
        self.lock = threading.Lock()
        self.eras = {}
        self.entry_offsets = [0]
        self.prefix_eras = {}
        self.lookup_table = []

    def collect(self, filename_treename):
        filename, treename = filename_treename

        with uproot.open({filename: treename}) as tree:
            branches = tree.values(recursive=True)

            era_key = f"{treename}\n" + "\n".join(
                f"{branch.name}\t{branch.interpretation.cache_key}"
                for branch in sorted(branches, key=lambda x: x.name)
            )

            num_entries = tree.num_entries

            branch_seek = []
            branch_bytes = []
            branch_offsets = []
            for branch in branches:
                num_baskets = branch.num_baskets
                branch_seek.append(branch.member("fBasketSeek")[:num_baskets])
                branch_bytes.append(branch.member("fBasketBytes")[:num_baskets])
                offsets = [0]
                for i in range(num_baskets):
                    _, stop = branch.basket_entry_start_stop(i)
                    offsets.append(stop)
                branch_offsets.append(offsets)

                assert offsets[-1] == num_entries

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
            prefix = "/".join(split_filename[:-1]) + "/"
            last_name = split_filename[-1]
            prefix_era = self.prefix_eras.get(prefix)
            if prefix_era is None:
                prefix_era = self.prefix_eras[prefix] = {
                    "index": len(self.prefix_eras),
                    "prefix": prefix,
                }

            self.lookup_table.append(
                {
                    "filename": last_name,
                    "prefix": prefix_era["index"],
                    "era": era["index"],
                    "tree": {
                        name: {
                            "seek": branch_seek[i],
                            "bytes": branch_bytes[i],
                            "offsets": branch_offsets[i],
                        }
                        for i, name in enumerate(era["names"])
                    },
                }
            )

    def final(self):
        with self.lock:
            return ak.Array(
                [
                    {
                        "era": [
                            {
                                "treename": x["treename"],
                                "names": x["names"],
                                "interpretations": x["interpretations"],
                            }
                            for x in self.eras.values()
                        ],
                        "offsets": self.entry_offsets,
                        "prefix": [x["prefix"] for x in self.prefix_eras.values()],
                        "file": self.lookup_table,
                    }
                ]
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

    array = collected_data.final()

    # upload_to_tiled(
    #     "http://127.0.0.1:8000?api_key=5b0076d3e33202d55884b2428a4f405e3ca84510c8c6b60fbe60d393998abbb0",
    #     "dataset_name",
    #     lookup_table,
    # )
