# LDA__Topics

Latent Dirichlet Allocation on a collection of **849 Reuters news articles**,
fitted two ways — **variational Bayes** (probabilistic, deterministic) and
**collapsed Gibbs sampling** (sampling-based) — and interpreted through top
words, relevance ranking, an intertopic distance map and per-word topic
assignments.

Built for **AML23702 Advance Natural Language Processing, Lab Exercise 4**.

**Runs in one command on a fresh clone**: every model output is committed, so
the app starts instantly without retraining.

---

## Pipeline

```
Reuters-21578 (8 categories, 849 single-label articles, frozen JSONL)
   │
   ├── 1. preprocess ── spaCy: lemmatise, keep NOUN/PROPN/ADJ/VERB, drop stopwords,
   │                    numbers, tickers and newswire boilerplate   → data/processed/tokens.jsonl
   ├── 2. DTM ───────── CountVectorizer, min_df=5, max_df=0.40      → 849 × 1,849, 97.5% sparse
   │                    90% train / 10% held out
   ├── 3. choose K ──── variational LDA for K ∈ {4..14} × 3 seeds   → k_sweep.csv
   ├── 4. fit K = 8 ─┬─ variational Bayes (scikit-learn, batch)
   │                 └─ collapsed Gibbs sampler (own implementation, 400 sweeps)
   ├── 5. evaluate ──── NPMI & UMass coherence, held-out perplexity, diversity,
   │                    NMI / purity vs the gold categories, VB↔Gibbs topic matching
   └── 6. interpret ─── labels, relevance (λ), intertopic map, representative documents
                                                                     → docs/topic_interpretation.md
```

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

To regenerate every artefact from the raw corpus (about 2½ minutes):

```bash
python scripts/run_pipeline.py              # preprocess → sweep → fit → evaluate → export
python scripts/make_interpretation.py       # docs/topic_interpretation.md
python scripts/check.py                     # self-check
```

Fixed seeds make every run byte-for-byte reproducible.

## The app

| Tab | Shows |
|---|---|
| Overview | corpus size, both models' quality metrics, one document before and after preprocessing |
| Topics | each topic's words as P(word \| topic), ranked by relevance with a λ slider; representative documents |
| Topic map | topics placed by MDS on Jensen-Shannon distance, sized by prevalence (as in LDAvis) |
| Documents | any article's topic mixture θ, and the Gibbs sampler's topic for every word, colour-coded |
| Choosing K | coherence (± SD over seeds), perplexity and NMI against K |
| Variational vs Gibbs | matched topics, Gibbs log-likelihood trace, topic × category heat map |
| Try your text | paste any text, get its topic mixture by folding it into the fitted model |

## Results (K = 8)

| Measure | Variational Bayes | Collapsed Gibbs |
|---|---|---|
| NPMI coherence ↑ | 0.223 | **0.229** |
| UMass coherence ↑ | **−1.429** | −1.497 |
| Held-out perplexity ↓ | 645.5 | **576.4** |
| Topic diversity ↑ | 0.875 | **0.900** |
| NMI vs categories ↑ | 0.584 | **0.602** |
| Purity ↑ | 0.727 | **0.733** |

Topics found by the Gibbs sampler (labels are the one human step, recorded in
`data/labels.json`):

| Topic | Label | Top words |
|---|---|---|
| T0 | OPEC & crude oil production | oil opec price crude barrel bpd saudi gulf production petroleum |
| T1 | General market commentary | market price rise fall low analyst high government demand increase |
| T2 | International trade & government | trade japan country foreign government united japanese minister import export |
| T3 | Interest rates & banking | rate bank interest cut prime day point money base credit |
| T4 | Gold mining & ports (mixed) | gold ounce ton south ship foot port ore reserve coin |
| T5 | Sugar trade & EC intervention | sugar tonne export trader production intervention import china output price |
| T6 | Coffee export quotas (ICO) | coffee quota export producer brazil ico meeting bag price talk |
| T7 | Corporate acquisitions | share offer buy stock pay acquire group unit exchange canada |

**What the numbers say.**

- Most categories come back as recognisable topics. In the Gibbs model, five
  topics are dominated by one category — interest 94%, sugar 90%, crude 84%,
  coffee 80%, acquisitions 74% of their documents — and trade by 61%.
  **Shipping gets no topic of its own in the Gibbs model**: ship articles are
  about Gulf tankers, grain cargoes and port strikes, so their words spread over
  the oil, gold/ports and trade topics. Variational Bayes does find a
  ports-and-strikes topic (61% ship), but pays for it by merging sugar with
  OPEC output.
- Gibbs spends one topic on **general market vocabulary** (*price, rise, fall,
  analyst*) shared by every commodity. That topic has no single category, yet
  it raises diversity and NMI, because it stops those words blurring the others.
- **Choosing K**: coherence rises steeply to K = 6 and is flat after it — every
  K from 6 to 14 is within one standard deviation of the best. Perplexity keeps
  falling as K grows, as it almost always does, so it cannot choose K. K = 8 is
  inside the plateau and equals the number of sampled categories, which lets the
  experiment test whether LDA recovers them.
- **The two inference methods agree on the clear topics and differ on the
  vague ones**: matched topics share 8–9 of their top-10 words for interest
  rates, coffee and sugar, but VB's ports/strikes and shares/gold topics have
  no Gibbs counterpart (one shared word each). Mean top-10 overlap is 5.75 / 10.
- The Gibbs log-likelihood climbs from −558,890 to −394,283 in the first 50
  sweeps, then changes by less than 0.3% over the 200 sampling sweeps after
  burn-in, so the averaged samples come from a chain that has settled.

## Layout

```
LDA__Topics/
├── app.py                  Streamlit interface (reads data/artifacts/ only)
├── lda_topics/
│   ├── config.py           every setting of the experiment
│   ├── corpus.py           frozen Reuters subset: build once, load always
│   ├── preprocess.py       spaCy lemmatisation + filters (regex fallback)
│   ├── dtm.py              document-term matrix, vocabulary pruning
│   ├── gibbs.py            collapsed Gibbs sampler for LDA
│   ├── models.py           variational LDA, Gibbs wrapper, fold-in, held-out perplexity
│   ├── metrics.py          NPMI, UMass, diversity, NMI/purity, JS topic matching
│   └── interpret.py        relevance, intertopic map, representative documents
├── data/
│   ├── corpus/reuters_subset.jsonl   849 raw articles with gold categories
│   ├── processed/tokens.jsonl        the preprocessed corpus
│   ├── labels.json                   human topic labels
│   └── artifacts/                    φ, θ, topic tables, sweep, trace, assignments, metrics
├── docs/topic_interpretation.md      the topic interpretation report
└── scripts/
    ├── run_pipeline.py     the whole experiment
    ├── make_interpretation.py
    ├── build_corpus.py     re-samples Reuters from NLTK (not needed - corpus is committed)
    └── check.py            self-check incl. a synthetic corpus with known topics
```

## Design notes

**Why two inference methods.** They approximate the same posterior from
opposite directions. Variational Bayes optimises a simpler distribution and is
deterministic; Gibbs samples the true posterior but is stochastic. Where they
agree, the topic is a property of the data; where they disagree, it is an
artefact of the method — that comparison is the most useful interpretation tool
here.

**Why a hand-written Gibbs sampler.** The update rule is the textbook equation
(Griffiths & Steyvers, 2004) and fits in twenty lines; using a library would
hide exactly the part the exercise is about. Plain Python lists beat NumPy for
K = 8, because NumPy's per-call overhead dominates on eight-element arrays — one
sweep over the 56,692 training tokens takes about 0.08 s.

**Why the same perplexity estimator for both.** scikit-learn's built-in
perplexity uses the variational bound, which cannot be computed for Gibbs.
Both models are scored by document completion (Wallach et al., 2009): θ is
folded in from half of each held-out document and the other half is scored.

**Why NPMI over perplexity.** Lower perplexity does not mean more readable
topics (Chang et al., 2009). NPMI correlates best with human judgements of
coherence (Röder et al., 2015), so it is the primary quality measure.

**Why lemmatise.** *price, prices, priced* would otherwise be three words with
a third of the counts each; none would reach a topic's top ten.

**Why the gold categories are kept.** They are never shown to the model; they
only serve to measure, after the fact, whether the unsupervised structure
matches the human one.

## Limitations

- Reuters contains near-duplicate wire items (the same EC sugar tender
  reported several times); they inflate some topics' representative lists.
- A Gibbs chain is one sample path; a second seed would give somewhat
  different topics for the vague themes.
- Bag-of-words ignores word order: *rate cut* and *cut rate* are the same.

## Data

Reuters-21578, Distribution 1.0 (ApteMod version as distributed with NLTK),
© Reuters Ltd and Carnegie Group, Inc.; freely available for research use.
