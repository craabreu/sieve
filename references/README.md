# Bibliography workflow

`references.bib` is a **generated artifact** and is gitignored. It is never hand-edited.
Every entry in it is either resolved from a DOI at build time, or drawn from a small,
explicitly quarantined set of DOI-less references.

## Usage

```sh
./generate_bibtex.sh            # build references.bib (uses .cache/)
./generate_bibtex.sh --refresh  # re-fetch every DOI, ignoring the cache
./generate_bibtex.sh --check    # validate only; write nothing (for CI / pre-commit)
```

The script exits nonzero and **writes nothing** if any check fails, so a broken
bibliography cannot be produced by a partially successful run.

## Adding a reference

1. Find its DOI.
2. Append `<citekey>  <DOI>` to `doi_list.txt`.
3. Run `./generate_bibtex.sh`.
4. Cite `<citekey>` from `wllr.md`.

If the work genuinely has **no** DOI (JMLR, NeurIPS, PMLR, and older proceedings often
have none), add it to `manual.bib` instead, following the rules in that file's header:
a publisher URL and a dated `verified against publisher record YYYY-MM-DD` note are
mandatory and mechanically enforced.

## Files

| File | Role | Tracked |
|---|---|---|
| `doi_list.txt` | `citekey → DOI` registry; the source of truth | yes |
| `manual.bib` | DOI-less entries, quarantined and dated | yes |
| `doi2bib.sh` | vendored resolver (`dx.doi.org` content negotiation) | yes |
| `generate_bibtex.sh` | build + validation | yes |
| `references.bib` | generated bibliography | **no** |
| `.cache/` | per-DOI resolver responses | **no** |

## What the validation actually prevents

The point of the workflow is that a reference which does not exist, or which is not the
one you meant, cannot reach `references.bib`. Each of these is enforced and tested:

| Failure | Caught by |
|---|---|
| DOI does not resolve | resolver output is not BibTeX → error |
| Network failure / HTML error page | same check (`doi2bib.sh` exits 0 on failure, so its output is treated as untrusted) |
| Typo'd DOI that resolves to a *different real paper* | returned `DOI={...}` is compared to the requested DOI |
| Corrupted or hand-edited cache file | cached content is validated on every run, not just on fetch |
| Duplicate citekey (silently drops a reference in BibTeX) | registry parse |
| Duplicate DOI under two citekeys | registry parse |
| Malformed registry line | registry parse |
| Hand-written entry sneaking into `manual.bib` without verification | missing `url` or dated note → error |
| A DOI-bearing entry hiding in `manual.bib` | `doi` field in `manual.bib` → error |
| **`wllr.md` citing a key that was never registered** | citation cross-check → error |
| Registered but never cited | warning (not an error — staged additions are legitimate) |

The citation cross-check is the one that catches a fabricated reference: a plausible-looking
`[Author2021Title]` in the manuscript that no one ever registered fails the build.

## Caveats

- **Citekey rewriting.** `doi2bib.sh` returns publisher-assigned keys (`Bremser_1978`,
  `Lehner_2023`). The generator rewrites these to the citekey from `doi_list.txt`, so
  regenerating never renames a key and breaks citations. It also avoids collisions —
  `Lehner_2023` and `Lehner_2024` would otherwise be one namespace away from clashing.
- **Crossref metadata is not infallible.** It is authoritative for the *existence* and
  identity of a work, and generally correct for volume/pages/year, but titles sometimes
  arrive with odd casing (`Hose — a novel substructure code`). Fix casing downstream in
  the LaTeX style, not by hand-editing the generated file.
- **The cross-check regex** matches the `[Key1234Suffix]` citation style used in `wllr.md`.
  If the manuscript moves to `\cite{...}`, update the pattern in `generate_bibtex.sh`.
- **Rate limiting.** Live fetches are spaced by `FETCH_DELAY` seconds (default 1). Raise it
  if `--refresh` on a long list starts failing.
