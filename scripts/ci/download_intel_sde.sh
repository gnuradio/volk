#!/bin/bash

set -e
set -x

function test_sde
{
    if ! [ -f ${SDE} ]; then
        echo "1"
    else
        ${SDE} -- ls > /dev/null
        echo $?
    fi
}

mkdir -p cache
cd cache

[ -z "${SDE_VERSION}" ] && SDE_VERSION=sde-external-8.35.0-2019-03-11-lin
[ -z "${SDE_URL1}" ] && SDE_URL1=https://software.intel.com/protected-download/267266/144917
[ -z "${SDE_URL2}" ] && SDE_URL2=https://software.intel.com/system/files/managed/32/db

SDE_TARBALL=${SDE_VERSION}.tar.bz2
SDE=$(pwd)/${SDE_VERSION}/sde64

if [ _$(test_sde) == _0 ]; then
    MSG="found working version: ${SDE_VERSION}"
else
    MSG="downloading: ${SDE_VERSION}"
    curl --verbose --form accept_license=1 --form form_id=intel_licensed_dls_step_1 \
         --output /dev/null --cookie-jar jar.txt \
         --location ${SDE_URL1}
    curl --verbose --cookie jar.txt --output ${SDE_TARBALL} \
         ${SDE_URL2}/${SDE_TARBALL}
    tar xvf ${SDE_TARBALL}
fi

echo $SDE
