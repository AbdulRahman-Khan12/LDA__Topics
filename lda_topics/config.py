"""Central configuration for LDA__Topics.

Every knob of the experiment lives here, so the numbers in the report can be
traced to one file. Changing anything here and re-running
``python scripts/run_pipeline.py`` regenerates every artefact.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_NAME = "LDA__Topics"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CORPUS_FILE = DATA_DIR / "corpus" / "reuters_subset.jsonl"     # frozen raw corpus
TOKENS_FILE = DATA_DIR / "processed" / "tokens.jsonl"          # preprocessed corpus
ARTIFACT_DIR = DATA_DIR / "artifacts"                            # model outputs
LABELS_FILE = DATA_DIR / "labels.json"                           # human topic labels
DOCS_DIR = PROJECT_ROOT / "docs"

# --- corpus --------------------------------------------------------------------
# Reuters-21578 (ApteMod split, as shipped with NLTK). Only single-label
# articles are kept, so every document has exactly one gold category to
# compare the unsupervised topics against.
CATEGORIES = ("acq", "coffee", "crude", "gold", "interest", "ship", "sugar", "trade")
DOCS_PER_CATEGORY = 110
MIN_WORDS = 60            # very short wire items ("shr 12 cts vs 10 cts") carry no topic
SEED = 42

# --- preprocessing ---------------------------------------------------------------
KEEP_POS = {"NOUN", "PROPN", "ADJ", "VERB"}
MIN_TOKEN_LEN = 3
# Newswire boilerplate: present in every category, so it can only blur topics.
DOMAIN_STOPWORDS = {
    "say", "said", "reuter", "reuters", "mln", "dlrs", "dlr", "pct", "cts", "year",
    "billion", "million", "also", "would", "could", "one", "two", "three", "last",
    "new", "company", "official", "tell", "make", "take", "week", "month", "today",
    "inc", "corp", "ltd", "co", "include", "expect", "add", "report", "early",
    "april", "march", "february", "january", "may", "june",
}

# --- document-term matrix ----------------------------------------------------------
MIN_DF = 5                # a word in fewer than 5 documents cannot define a topic
MAX_DF = 0.40             # a word in more than 40% of documents cannot separate them

# --- LDA ----------------------------------------------------------------------------
K = 8                     # = number of sampled categories; see the K sweep (k_sweep.csv)
K_SWEEP = (4, 6, 8, 10, 12, 14)
ALPHA = 0.1               # document-topic Dirichlet prior (sparse mixtures)
ETA = 0.01                # topic-word Dirichlet prior   (sparse topics)
VB_MAX_ITER = 100
GIBBS_ITERS = 400
GIBBS_BURN_IN = 200
GIBBS_THIN = 20           # average phi over every 20th sample after burn-in
TOP_N = 10                # words per topic used for coherence and display
RELEVANCE_LAMBDA = 0.6    # Sievert & Shirley (2014) suggest ~0.6
HOLDOUT_FRACTION = 0.1    # documents kept aside to measure perplexity
