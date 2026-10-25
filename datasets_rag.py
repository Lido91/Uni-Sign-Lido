"""Retrieval-augmented SLT data (RA-SFT, doc/signrag_style_retriever_plan.md §11).

``S2T_Dataset`` plus, for every clip, the top-K training sentences retrieved
for it (``retrieval_<split>_s2t.json`` written by signrag_retrieval.py; the
training-split file was retrieved with the self / identical-text /
same-video filters). They are rendered into one text segment returned as
``signsmd_text``, so the unchanged early-fusion model
``models_signsmd.Uni_Sign_SignSMD`` reads

    [retrieved sentences][translate prompt][pose]

--rag_mode real   : the retrieved sentences
           random : K random training sentences per clip (fixed per clip), a
                    control for "more tokens" vs "relevant content"
           empty  : no sentences (the baseline input plus one </s>)
           gloss  : the clip's ground-truth gloss sequence (oracle word-level
                    hints; CSL-Daily), a ceiling for word-level retrieval
           hint   : a precomputed hint string per clip (word-level retrieval,
                    word_retrieval_{h2s,csl}.py), read from the --rag_*_json
                    files {"results": [{"query": {"name": ...}, "hint": "..."}]}
Training applies retrieval dropout (SignRAG §3.2): with probability
--rag_dropout a sample gets the empty segment.
"""
import json
import random
import zlib

import utils
from config import train_label_paths
from datasets import S2T_Dataset

RAG_PREFIX = "Similar sentences: "
GLOSS_PREFIX = "Glosses: "


class S2T_Dataset_RAG(S2T_Dataset):
    def __init__(self, path, args, phase, retrieval_json):
        super().__init__(path, args, phase)
        self.mode, self.k = args.rag_mode, args.rag_k
        self.dropout = args.rag_dropout if phase == "train" else 0.0
        self.retrieved = {}
        if self.mode in ("real", "hint"):
            with open(retrieval_json) as f:
                results = json.load(f)["results"]
            if self.mode == "real":
                self.retrieved = {r["query"]["name"]: [t["text"] for t in r["topk"]] for r in results}
            else:
                self.retrieved = {r["query"]["name"]: r["hint"] for r in results}
            missing = [k for k in self.list if self.raw_data[k]["name"] not in self.retrieved]
            if missing:
                raise KeyError(f"{len(missing)} {phase} clips have no retrieval in {retrieval_json}, "
                               f"e.g. {missing[:3]}")
        elif self.mode == "random":
            self.pool = [v["text"] for v in utils.load_dataset_file(train_label_paths[args.dataset]).values()]

    def _sentences(self, name):
        if self.mode == "real":
            return self.retrieved[name][:self.k]
        rng = random.Random(zlib.crc32(name.encode()))       # same random sentences every epoch
        return rng.sample(self.pool, self.k)

    def render(self, name, gloss):
        if self.mode == "empty" or (self.dropout and random.random() < self.dropout):
            return ""
        if self.mode == "gloss":
            return GLOSS_PREFIX + gloss
        if self.mode == "hint":
            return self.retrieved[name]
        return RAG_PREFIX + " | ".join(self._sentences(name))

    def __getitem__(self, index):
        item = super().__getitem__(index)                    # name, pose, text, gloss, rgb
        return (*item, self.render(item[0], item[3]))

    def collate_fn(self, batch):
        descriptions = [item[-1] for item in batch]
        src_input, tgt_input = super().collate_fn([item[:-1] for item in batch])
        src_input["signsmd_text"] = descriptions
        return src_input, tgt_input
