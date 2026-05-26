# Editorial Agent PRs

Editorial agents may improve wording, structure, transitions, and clarity, but
they may not change claim scope, numeric results, generated summaries,
citations, or experiment interpretation in an editorial-only PR.

If an edit needs to change a claim, number, citation, generated block, or
interpretation, the agent must open an explicit claim change PR. Claim change PRs
must name the affected claim id, source artifact, and regenerated evidence path.

## Workflow

1. Generate or refresh evidence first.
2. Open a branch containing prose-only edits.
3. Keep generated files, tables, claim summaries, numbers, and citations
   unchanged unless the PR is explicitly marked as a claim change.
4. Select exactly one PR type in the PR template.
5. Run `make paper-ci`.
6. Let a human author review the diff and accept, reject, or revise the edits.

The review boundary is intentional: paper prose can be improved by an agent, but
empirical claims remain generated from repository artifacts and checked before
the PDF compiles.

## CI Contract

`make paper-ci` runs `the paper-support editorial gate` and then compiles the paper
through the normal generated-evidence path. On a pull request, the script reads
the PR type from the template checkbox, labels, or title prefix.

Editorial PRs fail CI if they touch generated paper artifacts, result artifacts,
claim registry files, source/test/script files, citation source files, numeric
text, citation-like references, or generated claim bindings.

Claim change PRs must fill in:

- **Claim id**
- **Artifact path**
- **Regenerated command**
- **Before/after scope**
- **Interpretation rationale**

The repository rule becomes meaningful only if the `Paper Gates` workflow is a
required status check on protected branches.

## Disclosure Language

AI agents were used as editorial and engineering assistants through reviewable
pull requests. Human authors accepted or rejected all edits. Empirical claims
were generated from repository artifacts and checked by CI before paper
compilation.
