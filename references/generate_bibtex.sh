#!/usr/bin/env bash
# ============================================================================
# generate_bibtex.sh — build references.bib from doi_list.txt + manual.bib
# ============================================================================
#
#   ./generate_bibtex.sh              build references.bib (uses cache)
#   ./generate_bibtex.sh --refresh    re-fetch everything, ignoring the cache
#   ./generate_bibtex.sh --check      validate registry + citations; no network
#                                     unless a DOI is not yet cached
#
# Design goal: it must be impossible for an unverified reference to reach
# references.bib. Every entry is either (a) fetched from dx.doi.org and
# validated to correspond to the requested DOI, or (b) quarantined in
# manual.bib under the rules documented in that file's header.
#
# NOTE ON ./doi2bib.sh: it exits 0 and prints an HTML error page when a DOI does
# not resolve. Its output is therefore treated as untrusted and validated here.
# ============================================================================

set -uo pipefail

cd "$(dirname "$0")" || exit 1

DOI_LIST="doi_list.txt"
MANUAL_BIB="manual.bib"
OUTPUT="references.bib"
CACHE_DIR=".cache"
FETCHER="./doi2bib.sh"
DOC="../wllr.md"

# Seconds to wait between live fetches, to stay polite to the DOI resolver.
FETCH_DELAY="${FETCH_DELAY:-1}"

REFRESH=0
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --refresh) REFRESH=1 ;;
    --check)   CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

errors=0
warnings=0
err()  { echo "  ERROR: $*" >&2; errors=$((errors + 1)); }
warn() { echo "  WARN:  $*" >&2; warnings=$((warnings + 1)); }

for f in "$DOI_LIST" "$MANUAL_BIB" "$FETCHER"; do
  [[ -e "$f" ]] || { echo "missing required file: $f" >&2; exit 1; }
done
[[ -x "$FETCHER" ]] || chmod +x "$FETCHER"
mkdir -p "$CACHE_DIR"

# ---------------------------------------------------------------------------
# 1. Parse and validate the registry
# ---------------------------------------------------------------------------
echo "==> Parsing $DOI_LIST"

keys=()
dois=()
while read -r key doi rest; do
  [[ -z "${key:-}" || "$key" == \#* ]] && continue
  if [[ -z "${doi:-}" ]]; then
    err "line for '$key' has no DOI"
    continue
  fi
  if [[ -n "${rest:-}" ]]; then
    err "$key: unexpected trailing text '$rest' (expected '<citekey> <DOI>')"
    continue
  fi
  if [[ "$doi" != 10.* ]]; then
    err "$key: '$doi' is not a DOI (must start with '10.')"
    continue
  fi
  if [[ ! "$key" =~ ^[A-Za-z][A-Za-z0-9:_-]*$ ]]; then
    err "$key: invalid citekey characters"
    continue
  fi
  keys+=("$key")
  dois+=("$doi")
done < "$DOI_LIST"

# Duplicate detection — a duplicate citekey silently drops a reference in BibTeX.
dupe_keys=$(printf '%s\n' "${keys[@]}" | sort | uniq -d)
[[ -n "$dupe_keys" ]] && err "duplicate citekeys: $(echo "$dupe_keys" | tr '\n' ' ')"
dupe_dois=$(printf '%s\n' "${dois[@]}" | tr 'A-Z' 'a-z' | sort | uniq -d)
[[ -n "$dupe_dois" ]] && err "duplicate DOIs: $(echo "$dupe_dois" | tr '\n' ' ')"

echo "    ${#keys[@]} DOI entries"

# ---------------------------------------------------------------------------
# 2. Validate manual.bib (the only hand-written BibTeX)
# ---------------------------------------------------------------------------
echo "==> Validating $MANUAL_BIB"

manual_keys=($(grep -oE '^@[A-Za-z]+\{[^,]+,' "$MANUAL_BIB" | sed -E 's/^@[A-Za-z]+\{//; s/,$//'))
echo "    ${#manual_keys[@]} manual entries"

# Rules 2 and 3 from the manual.bib header: every entry needs a publisher url
# and a dated verification note. Entries are delimited by brace depth, not by
# lines, since fields wrap across lines.
manual_problems=$(awk '
  # Strip whole-line comments so the % header cannot trip the parser.
  /^[[:space:]]*%/ { next }
  {
    line = $0
    if (depth == 0 && match(line, /^@[A-Za-z]+\{/)) {
      inentry = 1; key = line
      sub(/^@[A-Za-z]+\{/, "", key); sub(/,.*$/, "", key)
      gsub(/[[:space:]]/, "", key)
      buf = ""
    }
    if (inentry) {
      buf = buf "\n" line
      n = gsub(/\{/, "{", line); m = gsub(/\}/, "}", line)
      depth += n - m
      if (depth <= 0) {
        if (buf !~ /url[[:space:]]*=/)
          print key "\tmissing '\''url'\'' (publisher record required)"
        if (buf !~ /note[[:space:]]*=[^}]*verified against publisher record [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/)
          print key "\tmissing dated '\''verified against publisher record YYYY-MM-DD'\'' note"
        if (buf ~ /[^a-zA-Z]doi[[:space:]]*=/)
          print key "\thas a '\''doi'\'' field — move it to the DOI list"
        inentry = 0; depth = 0
      }
    }
  }
' "$MANUAL_BIB")

if [[ -n "$manual_problems" ]]; then
  while IFS=$'\t' read -r ekey msg; do
    [[ -z "$ekey" ]] && continue
    err "manual.bib/$ekey: $msg"
  done <<< "$manual_problems"
fi

# Cross-file duplicate keys
if [[ ${#manual_keys[@]} -gt 0 && ${#keys[@]} -gt 0 ]]; then
  overlap=$(comm -12 \
    <(printf '%s\n' "${keys[@]}" | sort) \
    <(printf '%s\n' "${manual_keys[@]}" | sort))
  [[ -n "$overlap" ]] && err "citekey defined in BOTH files: $(echo "$overlap" | tr '\n' ' ')"
fi

# ---------------------------------------------------------------------------
# 3. Fetch + validate each DOI
# ---------------------------------------------------------------------------
echo "==> Resolving DOIs"

tmp_out=$(mktemp)
trap 'rm -f "$tmp_out"' EXIT

fetched=0
cached=0

for i in "${!keys[@]}"; do
  key="${keys[$i]}"
  doi="${dois[$i]}"
  # Cache filename: DOI with '/' and other awkward chars flattened.
  cache_file="$CACHE_DIR/$(printf '%s' "$doi" | tr '/:()' '____').bib"

  from_cache=0
  if [[ $REFRESH -eq 0 && -s "$cache_file" ]]; then
    raw=$(cat "$cache_file")
    from_cache=1
    cached=$((cached + 1))
  else
    [[ $fetched -gt 0 ]] && sleep "$FETCH_DELAY"
    raw=$("$FETCHER" "$doi" 2>/dev/null)
    fetched=$((fetched + 1))
  fi

  # --- Validation ---------------------------------------------------------
  # Applied to cached content as well as freshly fetched content: a truncated
  # or hand-edited cache file must not be able to reach references.bib.
  # ./doi2bib.sh exits 0 and prints an HTML error page on failure, so its output
  # is treated as untrusted.
  if [[ -z "$raw" ]]; then
    err "$key ($doi): empty record${from_cache:+ (cached)}"
    continue
  fi
  if [[ "$(printf '%s' "$raw" | sed 's/^[[:space:]]*//' | cut -c1)" != "@" ]]; then
    err "$key ($doi): not BibTeX — DOI not found, network failure, or HTML error page"
    [[ $from_cache -eq 1 ]] && err "  ^ from cache; delete $cache_file and retry"
    continue
  fi
  # The record must actually be the DOI we asked for. This catches a typo that
  # happens to resolve to a *different real paper* — which a content-only check
  # would happily accept, and which is the subtlest way a wrong reference lands
  # in the bibliography.
  got_doi=$(printf '%s' "$raw" | grep -oiE 'DOI=\{[^}]+\}' | head -1 | sed -E 's/^DOI=\{//I; s/\}$//')
  if [[ -z "$got_doi" ]]; then
    err "$key ($doi): record has no DOI field"
    continue
  fi
  if [[ "$(printf '%s' "$got_doi" | tr 'A-Z' 'a-z')" != "$(printf '%s' "$doi" | tr 'A-Z' 'a-z')" ]]; then
    err "$key: requested DOI '$doi' but record is for '$got_doi'"
    continue
  fi

  [[ $from_cache -eq 0 ]] && printf '%s\n' "$raw" > "$cache_file"

  # Rewrite the publisher's citekey (e.g. "Bremser_1978") to our stable one.
  # Without this, regenerating renames keys and breaks every citation.
  printf '%s\n' "$raw" \
    | sed -E "1s/^[[:space:]]*@([A-Za-z]+)\{[^,]*,/@\1{$key,/" >> "$tmp_out"
  printf '\n' >> "$tmp_out"
done

echo "    $cached cached, $fetched fetched"

# ---------------------------------------------------------------------------
# 4. Cross-check citations in the manuscript (heuristic, warning-level)
# ---------------------------------------------------------------------------
if [[ -f "$DOC" ]]; then
  echo "==> Cross-checking citations in $DOC"
  defined=$(printf '%s\n' "${keys[@]}" "${manual_keys[@]}" | sort -u)
  # Matches the [Key1234Suffix] citation style used in wllr.md.
  cited=$(grep -oE '\[[A-Z][A-Za-z0-9]*[0-9]{4}[A-Za-z0-9]*\]' "$DOC" \
          | tr -d '[]' | sort -u)

  while IFS= read -r c; do
    [[ -z "$c" ]] && continue
    grep -qx "$c" <<< "$defined" || err "$DOC cites [$c], which is not defined in $DOI_LIST or $MANUAL_BIB"
  done <<< "$cited"

  while IFS= read -r d; do
    [[ -z "$d" ]] && continue
    grep -qx "$d" <<< "$cited" || warn "$d is registered but never cited in $DOC"
  done <<< "$defined"
fi

# ---------------------------------------------------------------------------
# 5. Emit — only if everything passed
# ---------------------------------------------------------------------------
if [[ $errors -gt 0 ]]; then
  echo ""
  echo "FAILED: $errors error(s); $OUTPUT was NOT written." >&2
  exit 1
fi

if [[ $CHECK_ONLY -eq 1 ]]; then
  echo ""
  echo "OK: registry valid, $((${#keys[@]} + ${#manual_keys[@]})) references, $warnings warning(s). (--check: no output written)"
  exit 0
fi

{
  echo "% ==========================================================================="
  echo "% references.bib — GENERATED FILE, DO NOT EDIT"
  echo "% ==========================================================================="
  echo "%"
  echo "% Regenerate with:  references/generate_bibtex.sh"
  echo "%"
  echo "% Sources:"
  echo "%   doi_list.txt  ${#keys[@]} entries resolved via dx.doi.org and validated"
  echo "%                 to match the requested DOI"
  echo "%   manual.bib    ${#manual_keys[@]} entries with no registered DOI, verified"
  echo "%                 against publisher records"
  echo "%"
  echo "% Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "% ==========================================================================="
  echo ""
  cat "$tmp_out"
  echo ""
  sed 's/^%.*$//' "$MANUAL_BIB" | sed '/^$/N;/^\n$/D'
} > "$OUTPUT"

echo ""
echo "OK: wrote $OUTPUT — $((${#keys[@]} + ${#manual_keys[@]})) references, $warnings warning(s)."
