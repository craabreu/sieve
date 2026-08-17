#!/bin/zsh
if [[ $# -ne 1 ]]; then
  echo "usage: doi2bib.sh <DOI>"
  exit
fi
bib=$(curl -sLH "Accept: application/x-bibtex;q=1" http://dx.doi.org/$1 | sed '/^\s*$/d')
echo "$bib"
# if [[ "${bib[1,1]}" == "@" ]]; then
#   echo "$bib" | sed -E '/url\s*=\s*\{/d'
# else
#   echo "Error: DOI not found"
# fi
