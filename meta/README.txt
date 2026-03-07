Place Re-DocRED metadata files here.

Required for official evaluation:
  rel2id.json   — relation-to-ID mapping from the Re-DocRED dataset
                  (download from https://github.com/tonytan48/Re-DocRED)

The official evaluation script (evaluation.py) reads this file at:
  meta/rel2id.json

If rel2id.json is not available, the evaluate_extraction.py converter
falls back to the REL_NAME_TO_ID mapping defined in prompts.py.
