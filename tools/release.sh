#!/usr/bin/env bash
setopt ERR_EXIT #exit on error
#Project name
project=volk
#What to prefix in release tags in front of the version number
releaseprefix="v"
#Remote to push to
remote="origin"
#Name of the Changelog file
changelog="CHANGELOG.md"
#Name of the file where the last release tag is stored
lastreleasefile=".lastrelease"

#use your $EDITOR, unless unset, in which case: do the obvious
EDITOR="${EDITOR:=vim}"

tempdir="$(mktemp -d)"
deltafile="${tempdir}/delta.md"
annotationfile="${tempdir}/tag.rst"

# if using BSD signify:
pubkey="${HOME}/.signify/${project}-signing-001.pub"
seckey="${HOME}/.signify/${project}-signing-001.sec"

#use parallel pigz if available, else gzip
gz=$(which pigz 2> /dev/null || which gzip)

# uncomment the following lines if using GPG
# echo "Will use the following key for signing…"
# signingkey=$(git config user.signingkey)
# gpg2 --list-keys "${signingkey}" || echo "Can't get info about key ${signingkey}.  Did you forget to do 'git config --local user.signingkey=0xDEADBEEF'?'"
# echo "… end of key info."

# 1. Get the version number from CMake file
version_major=$(grep -i 'set(version_info_major' CMakeLists.txt |\
                    sed 's/.*VERSION_INFO_MAJOR_VERSION[[:space:]]*\([[:digit:]a-zA-z-]*\))/\1/i')
version_minor=$(grep -i 'set(version_info_minor' CMakeLists.txt |\
                    sed 's/.*VERSION_INFO_MINOR_VERSION[[:space:]]*\([[:digit:]a-zA-z-]*\))/\1/i')
version_maint=$(grep -i 'set(version_info_maint' CMakeLists.txt |\
                    sed 's/.*VERSION_INFO_MAINT_VERSION[[:space:]]*\([[:digit:]a-zA-z-]*\))/\1/i')
version="${version_major}.${version_minor}.${version_maint}"
last_release="$(cat ${lastreleasefile})"
echo "Releasing version ${version}"

# 2. Prepare Changelog
echo "appending git shortlog to CHANGELOG:"
shortlog="
## [${version}] - $(date +'%Y-%m-%d')

$(git shortlog -e ${last_release}..HEAD)
"
echo "${shortlog}"

echo "${shortlog}" > ${deltafile}

read -q "editcl?Do you want to edit the Changelog addtions (${deltafile}) using ${EDITOR}? (y/n)" || echo "proceeding unedited"
if [ "${editcl}" = "y" ]; then
    ${EDITOR} ${deltafile}
fi
echo "\n$(cat ${deltafile})" >> ${changelog}
echo "${releaseprefix}${version}" > ${lastreleasefile}

# 3. Commit changelog
git commit -m "Release ${version}" "${changelog}" "${lastreleasefile}" CMakeLists.txt

# 4. prepare tag
cat "${deltafile}" > ${annotationfile}
# Append the HEAD commit hash to the annotation
echo "git-describes-hash: $(git rev-parse --verify HEAD)" >> "${annotationfile}"

if type 'signify-openbsd' > /dev/null; then
    signaturefile="${tempdir}/annotationfile.sig"
    signify-openbsd -S -x "${signaturefile}" -s "${seckey}" -m "${annotationfile}"
    echo "-----BEGIN SIGNIFY SIGNATURE-----" >> "${annotationfile}"
    cat "${signaturefile}" >> "${annotationfile}"
    echo "-----END SIGNIFY SIGNATURE-----" >> "${annotationfile}"
fi

# finally tag the release and sign it
## add --sign to sign using GPG
git tag --annotate --cleanup=verbatim -F "${annotationfile}" "${releaseprefix}${version}"

# 5. Create archive
tarprefix="${project}-${version}"
outfile="${tempdir}/${tarprefix}.tar"
git archive "--output=${outfile}" "--prefix=${tarprefix}/" HEAD
echo "Created tape archive '${outfile}' of size $(du -h ${outfile})."

# 6. compress
echo  "compressing:"
echo  "gzip…"
${gz} --keep --best "${outfile}"
#--threads=0: guess number of CPU cores
echo "xz…"
xz --keep -9 --threads=0 "${outfile}"
# echo "zstd…"
# zstd --threads=0 -18 "${outfile}"
echo "…compressed."

# 7. sign
echo "signing file list…"
filelist="${tempdir}/${version}.sha256"
pushd "${tempdir}"
sha256sum --tag *.tar.* > "${filelist}"
signify-openbsd -S -e -s "${seckey}" -m "${filelist}"
echo "…signed. Check with 'signify -C -p \"${pubkey}\" -x \"${filelist}\"'."
signify-openbsd -C -p "${pubkey}" -x "${filelist}.sig"
popd
echo "checked."

#8. bundle archives
mkdir -p archives
cp "${tempdir}"/*.tar.* "${filelist}.sig" "${pubkey}" archives/
echo "Results can be found under $(pwd)/archives"

#9. Push to origin
read -q "push?Do you want to push to origin? (y/n)" || echo "not pushing"
if [ "${push}" = "y" ]; then
    git push "${remote}" HEAD
    git push "${remote}" "v${releaseprefix}${version}"
fi
