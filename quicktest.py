from __future__ import annotations

import tempfile

import uproot
from tiled.catalog import in_memory
from tiled.client import Context, from_context
from tiled.server.app import build_app

import tiled_uproot

files = [
    {
        "/home/jpivarski/storage/data/Run2018D-DoubleMuon-Nano25Oct2019_ver2-v1-974F28EE-0FCE-4940-92B5-870859F880B1.root": "Events"
        # "/home/jpivarski/storage/data/Run2012BC_DoubleMuParked_Muons.root": "Events"
    }
]
files = uproot._util.regularize_files(files, steps_allowed=False)

collected_data = tiled_uproot.populate.CollectedData()
for filename_treename in files:
    collected_data.collect(filename_treename)

root_metadata = collected_data.to_array()

# Build an app equivalent to `tiled serve catalog --temp`
tmpdir = tempfile.TemporaryDirectory()
catalog = in_memory(writable_storage=tmpdir.name)
app = build_app(catalog)

# Run the app event loop on a background thread.
context = Context.from_app(app)
client = from_context(context)

# Write to the database
client.write_awkward(root_metadata, key="root_metadata")

# Read from the database
awkward_client = client["root_metadata"].read()

tree = tiled_uproot.extract.TiledUproot("root_metadata", awkward_client)

tree.arrays(["nMuon", "Muon_pt"], entry_start=100, entry_stop=10000).show(type=True)
