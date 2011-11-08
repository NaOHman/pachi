#!/bin/sh
#
# test-game: Run testcases within a single game
#
# Usage: test-game FILENAME
#
# Examine a game record and run testcases for all moves described
# in the "game comment".

sgf="$1"

sed -ne '/GC\[/,/\]/{ s/.*GC\[//; s/\].*//; p; }' <"$sgf" |
	while read moven cases desc; do
		echo "Examining move $moven"; sleep 1
		tools/sgf2gtp.pl -g -n $((moven-1)) <"$sgf" | ./pachi -t =20000
		echo "Testcases: $cases"
		echo "Confirm and press enter..."; read xx </dev/tty
	done