#!/usr/bin/env bash
while IFS="" read -r repo; do
    url="$(echo $repo | cut -d: -f-1,2)"
    sha="$(echo $repo | cut -d: -f4)"
    name="$(echo $url | cut -d "/" -f 5 | cut -d "." -f 1)"

    if [[ ! -d "Repos/$name" ]]; then
        mkdir -p "Repos/$name"
        (cd "Repos/$name" &&
            git init &&
            git remote add origin "$url" &&
            GIT_TERMINAL_PROMPT=0 git fetch --depth 1 origin "$sha" &&
            git checkout FETCH_HEAD)
    else
        echo "$name was already cloned, continuing"
    fi
done <github_urls.txt
