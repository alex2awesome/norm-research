# Transcript-clean repair labeling

This is an isolation repair pass. Use only the assigned chunk, its frozen bank,
and the labeling guides named in the prompt. Do not inspect any other repository
file or directory, and do not search for prior labels, proposals, audits, truth,
or earlier transcripts.

For reading the allowed files, use only direct, explicitly targeted commands such
as `sed -n`, `head`, `tail`, or `jq` with the allowed file path in that same
command. Prefer `jq` selectors that perform any needed filtering themselves.

Do not use Python or any other general-purpose interpreter. Do not use `ls`,
`find`, `git`, recursive search, network commands, or command pipelines containing
`rg`/`grep`. Do not use shell variables or path discovery. If a permitted direct
read is insufficient, read another explicit range from the same allowed file.

Label every assigned item independently from scratch. A typed abstention is a
valid outcome; never force a leaf.
