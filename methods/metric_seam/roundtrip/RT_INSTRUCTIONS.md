# Blind rule-compilation crew (code round-trip experiment, 2026-08-12)

You are compiling prose scoring rules into Python. You see ONLY each rule's text. Do not
read any other files in the repository; do not speculate about where the rules came from.

Read your assigned input_rt_c<k>.json: a list of {job_id, rule}. For EACH rule, write a
pure-Python function that scores a document string on the 0-10 scale THE RULE DESCRIBES.

Requirements:
- def score(text: str) -> float   (0..10; deterministic; no imports beyond re, math,
  statistics, string, collections; no network, no files, no LLM calls)
- Implement the rule AS WRITTEN, as faithfully as its prose allows. Where the rule names
  qualitative properties with no procedure, approximate them with the best deterministic
  text-computable proxy the rule's own wording licenses (word/sentence statistics, keyword
  and pattern matches, counts, structure). Never return a constant unless the rule is
  genuinely constant.
- Handle edge cases (empty text, very long text) without exceptions.

Output: write output_rt_c<k>.py in this same directory, of the form:

    # AUTO: blind rule compilation chunk c<k>
    def score__<job_id>(text):
        ...

one function per job_id (double underscore between 'score' and the job_id; job_id used
verbatim). At the end include: JOB_IDS = [...] listing every job_id you implemented.
Final reply: just the count of functions written.
