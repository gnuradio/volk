#!/bin/bash
# Script to check the list of git submitters against the table of re-submitting
# users from the AUTHORFILE. Requires the authors to be listed in
# | ... | ... | email@address.com |
# format.
#
# We can add another table of "git committers who are exempt from the need to
# relicense due to their contributions being under an acceptable license
# already" if we need; no changes to this script would be necessary.
#
# This script is part of VOLK.
#
# Copyright 2021 Marcus Müller
# SPDX-License-Identifier: MPL-2.0

rootdir=`git rev-parse --show-toplevel`
if [[ "$#" -lt 1 ]]
then
    authorfile=$rootdir/AUTHORS_RESUBMITTING_UNDER_LGPL_LICENSE.md
else
    authorfile=$1
fi
if [[ ! -r $authorfile ]]
then
    echo "$authorfile: file not readable"
    exit -1
fi

allfiles=`git ls-files $rootdir`
lgplers="$(sed -ne 's/^|[^|]*|[^|]*| \([^|]*\)|/\1/ip' $authorfile)"
lgplers="$lgplers 32478819+fritterhoff@users.noreply.github.com douggeiger@users.noreply.github.com"
authorcounts="$(echo "$allfiles" | while read f; do git blame --line-porcelain --ignore-rev 092a59997a1e1d5f421a0a5f87ee655ad173b93f $f 2>/dev/null | sed -ne 's/^author-mail <\([^>]*\)>/\1/p'; done | sort -f | uniq -ic | sort -n)"

total_loc=0
missing_loc=0

while read -r line
do
    authoremail=$(echo "$line" | sed 's/^ *\([[:digit:]]*\) *\([^, ]*\)$/\2/g')
    authorlines=$(echo "$line" | sed 's/^ *\([[:digit:]]*\) *\([^, ]*\)$/\1/g')
    total_loc=$(( $authorlines + $total_loc ))
    if ! ( echo "$lgplers" | grep -i "$authoremail" ) > /dev/null
    then
        echo "missing: \"$authoremail\" (${authorlines} LOC)"
        missingloc=$(($missingloc + $authorlines))
    fi
done < <(echo "$authorcounts")

percentage=$(echo "scale=2; 100.0 * $missingloc/$total_loc" | bc)
echo "Missing $missingloc of $total_loc LOC in total ($percentage%)"

if [[  "$missingloc" -gt 0 ]]
then
   exit -2
fi
exit 0
