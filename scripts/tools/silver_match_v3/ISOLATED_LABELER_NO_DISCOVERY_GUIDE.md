# Transcript-safe file access

This is an operational isolation rule only; it does not change any labeling
boundary or decision definition in the independent-labeling guide.

- Read only the exact guide, bank, and assigned chunk named in the prompt.
- Do not run discovery or search commands, including `ls`, `find`, `fd`, `rg`,
  `grep`, `git`, or shell glob searches, even against an allowed file.
- Do not use a network command or a general-purpose interpreter.
- Read the complete allowed files directly with `sed -n` (using a sufficiently
  large line range). You may use additional `sed -n` ranges on those same exact
  files if needed.
- Do not inspect any other file or directory. Return the labels directly after
  reading the allowed inputs.
