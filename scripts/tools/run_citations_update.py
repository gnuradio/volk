#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Johannes Demel.
#
# This file is part of VOLK
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

import argparse
from pprint import pprint
import regex
import json
import pathlib


def parse_name(contributor):
    name_pattern = "(.*?) <"
    name = regex.search(name_pattern, contributor).group().replace(" <", "")
    email_pattern = "<(.*?)>"
    email = regex.search(email_pattern, contributor).group().replace(
        "<", "").replace(">", "")
    return name, email


def parse_contributors(contributors):
    result = []
    for c in contributors:
        name, email = parse_name(c)
        result.append({"name": name, "email": email})
    return result


name_aliases = {
    'alesha72003': "Alexey Slokva",
    'dernasherbrezon': "Andrey Rodionov",
    'Doug': "Douglas Geiger",
    'Doug Geiger': "Douglas Geiger",
    "Federico 'Larroca' La Rocca": "Federico Larroca",
    'git-artes': "Federico 'Larroca' La Rocca",
    'ghostop14': "Mike Piscopo",
    'gnieboer': "Geof Nieboer",
    'Jam Quiceno': "Jam M. Hernandez Quiceno",
    'jdemel': "Johannes Demel",
    'Marc L': "Marc Lichtman",
    'Marcus Mueller': "Marcus Müller",
    'Michael L Dickens': "Michael Dickens",
    'Micheal Dickens': "Michael Dickens",
    'namccart': "Nicholas McCarthy",
    'hcab14': "Christoph Mayer",
    'cmayer': "Christoph Mayer",
    'root': "Philip Balister",
    'jsallay': "John Sallay"}


def fix_known_names(contributors):
    for c in contributors:
        c['name'] = name_aliases.get(c['name'], c['name'])
    return contributors


def merge_names(contributors):
    results = []
    names = sorted(list(set([c['name'] for c in contributors])))
    for name in names:
        emails = []
        for c in contributors:
            if name == c['name']:
                emails.append(c['email'])
        results.append({'name': name, 'email': emails})
    return results


def normalize_names(contributors):
    for c in contributors:
        name = c['name'].split(' ')
        if len(name) > 1:
            name = f'{name[-1]}, {" ".join(name[0:-1])}'
        else:
            name = name[0]
        c['name'] = name

    return contributors


def find_citation_file(filename='.zenodo.json'):
    # careful! This function makes quite specific folder structure assumptions!
    file_loc = pathlib.Path(__file__).absolute()
    file_loc = file_loc.parent
    citation_file = next(file_loc.glob(f'../../{filename}'))
    return citation_file.resolve()


def load_citation_file(filename):
    with open(filename, 'r') as file:
        citation_file = json.load(file)
    return citation_file


def update_citation_file(filename, citation_data):
    with open(filename, 'w')as file:
        json.dump(citation_data, file, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Update citation file')
    parser.add_argument('contributors', metavar='N', type=str, nargs='+',
                        help='contributors with emails: Name <email>')
    args = parser.parse_args()

    contributors = args.contributors[0].split("\n")
    contributors = parse_contributors(contributors)
    contributors = fix_known_names(contributors)
    contributors = merge_names(contributors)
    contributors = normalize_names(contributors)

    citation_file_name = find_citation_file()
    citation_file = load_citation_file(citation_file_name)

    creators = citation_file['creators']

    git_names = sorted([c['name'] for c in contributors])
    cite_names = sorted([c['name'] for c in creators])
    git_only_names = []
    for n in git_names:
        if n not in cite_names:
            git_only_names.append(n)

    for name in git_only_names:
        creators.append({'name': name})

    # make sure all contributors are sorted alphabetically by their family name.
    creators = sorted(creators, key=lambda x: x['name'])
    maintainers = ["Demel, Johannes", "Dickens, Michael"]
    maintainer_list = list(
        filter(lambda x: x['name'] in maintainers, creators))
    creators = list(filter(lambda x: x['name'] not in maintainers, creators))
    nick_list = list(filter(lambda x: ', ' not in x['name'], creators))
    fullname_list = list(filter(lambda x: ', ' in x['name'], creators))

    creators = maintainer_list + fullname_list + nick_list

    citation_file['creators'] = creators
    update_citation_file(citation_file_name, citation_file)


if __name__ == "__main__":
    main()
