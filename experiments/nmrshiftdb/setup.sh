#!/usr/bin/env bash
# Build the NMRShiftDB code generator that predictors/nmrshiftdb.py calls.
#
# Fetches, at revision 2699 of https://sourceforge.net/p/nmrshiftdb2/code/,
# the jars NMRShiftDB ships (its CDK, version 2.4-SNAPSHOT) and its
# ExtendedHOSECodeGenerator, installs a JDK with conda, and compiles both with
# NmrshiftdbCodes.java into experiments/external/nmrshiftdb2-r2699 (ignored by
# git, like the DASH-tree checkout). NMRShiftDB is AGPL-3.0, so none of it is
# copied into this repository.
#
# Usage: experiments/nmrshiftdb/setup.sh [target-dir]
# Needs `svn` and `conda` (or `mamba`) on PATH.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$HERE/../external/nmrshiftdb2-r2699}"
REV=2699
URL=https://svn.code.sf.net/p/nmrshiftdb2/code/trunk/nmrshiftdb2

mkdir -p "$TARGET"
cd "$TARGET"
[ -d lib ] || svn export -q -r "$REV" "$URL/lib" lib
[ -f src/org/openscience/nmrshiftdb/util/ExtendedHOSECodeGenerator.java ] || {
  mkdir -p src/org/openscience/nmrshiftdb/util
  svn export -q -r "$REV" \
    "$URL/src/java/org/openscience/nmrshiftdb/util/ExtendedHOSECodeGenerator.java" \
    src/org/openscience/nmrshiftdb/util/ExtendedHOSECodeGenerator.java
}
# The extended generator calls one method of NMRShiftDB's AtomUtils,
# isTetrahedral, whose class otherwise pulls in the database layer. The
# method is copied verbatim from that revision into a class of its own.
UTIL=src/org/openscience/nmrshiftdb/util
[ -f "$UTIL/AtomUtils.java" ] || {
  svn cat -r "$REV" "$URL/src/java/org/openscience/nmrshiftdb/util/AtomUtils.java" > AtomUtils.full
  {
    echo "package org.openscience.nmrshiftdb.util;"
    echo "import java.util.List;"
    echo "import org.openscience.cdk.CDKConstants;"
    echo "import org.openscience.cdk.geometry.BondTools;"
    echo "import org.openscience.cdk.interfaces.IAtom;"
    echo "import org.openscience.cdk.interfaces.IAtomContainer;"
    echo "import org.openscience.cdk.interfaces.IBond;"
    echo "/** AtomUtils.isTetrahedral of NMRShiftDB revision $REV, verbatim. */"
    echo "public class AtomUtils {"
    awk '/public static int isTetrahedral\(/{on=1} on{print} on && /^\t}/{exit}' AtomUtils.full
    echo "}"
  } > "$UTIL/AtomUtils.java"
  rm AtomUtils.full
}
[ -x jdk/bin/javac ] || conda create -y -q -p "$TARGET/jdk" -c conda-forge openjdk=17 >/dev/null

CP="$(ls "$TARGET"/lib/*.jar | tr '\n' ':')"
rm -rf classes
mkdir classes
jdk/bin/javac -nowarn -encoding UTF-8 -cp "$CP" -d classes \
  src/org/openscience/nmrshiftdb/util/ExtendedHOSECodeGenerator.java \
  src/org/openscience/nmrshiftdb/util/AtomUtils.java \
  "$HERE/NmrshiftdbCodes.java"
echo "built $TARGET/classes"
