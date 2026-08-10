# #HashtagWars — SemEval-2017 Task 6 ("Learning a Sense of Humor")

Humor-verdict dataset of tweets submitted to the "Hashtag Wars" segment of the
Comedy Central show **@midnight**. For each episode's hashtag prompt, the show
staff picked the 10 funniest tweets and one overall winner — these editorial
picks are the verdicts.

## Provenance
- Task site: https://alt.qcri.org/semeval2017/task6/ (original "Data and Tools"
  download links are no longer directly listed on the page).
- Data obtained 2026-07-28 from the GitHub mirror
  https://github.com/cbaziotis/datastories-semeval2017-task6 (master tarball),
  which vendored the official `train_dir` / `trial_dir` / `evaluation_dir`
  releases from the task organizers (Potash, Romanov & Rumshisky, UMass Lowell
  Text Machine Lab). Task paper: Potash et al., SemEval-2017 Task 6
  (https://aclanthology.org/S17-2004/).
- Original per-split READMEs preserved as `README_train_orig.txt` and
  `README_eval_orig.txt`.

## License notes
- No explicit license was distributed with the SemEval data; SemEval datasets
  are conventionally released for research use. Tweets remain the property of
  their authors and Twitter/X ToS applies. The mirror repo (code) is the
  DataStories team's; the data files carry no separate license. Treat as
  research-use-only; do not redistribute publicly without checking with the
  task organizers.

## Layout & verdict structure
One TSV file per hashtag (filename = hashtag, e.g. `Cereal_Songs.tsv`), with
tab-separated columns:

```
tweet_id \t tweet_text \t label
```

Labels (picked by @midnight show staff):
- `0` = tweet not selected into the episode's top-10
- `1` = in the top-10 but not the winner
- `2` = the winning tweet of the episode

`evaluation_data/` files have no label column (official blind eval set);
`gold_labels/` holds the same 6 hashtags with labels.

## Counts (verified after download)
| split | hashtag files | tweets | label 0 | label 1 | label 2 |
|---|---|---|---|---|---|
| train_data | 101 | 11,325 | 10,326 | 898 | 101 |
| trial_data | 5 | 660 | 610 | 45 | 5 |
| gold_labels | 6 | 749 | 689 | 54 | 6 |
| evaluation_data (unlabeled) | 6 | 749 | — | — | — |
| **total labeled** | **112** | **12,734** | 11,625 | 997 | 112 |

## Usage
`load.py` yields `(hashtag, tweet_text, label)`:

```python
from load import load
for hashtag, text, label in load("train"):
    ...
```
