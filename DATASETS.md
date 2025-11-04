# Datasets

Preprocessed datasets are hosted on Zenodo.

Please download and unpack into `datasets/<key>/` with the following layout:

```
datasets/<key>/
  <key>_network.txt                 # NCOL undirected edgelist (int IDs)
  <key>_communities.txt             # one community per line (tab-separated)
  <key>_node_to_communities.csv     # node_id,comm1,comm2,...
```

Replace `<key>` with: `dblp`, `lj`, `youtube`, `so`,  `amazon`,  `deezer`,  `twitch`,  `github`, `text`, or `LFR_*`.
