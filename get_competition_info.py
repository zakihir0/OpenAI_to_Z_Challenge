#!/usr/bin/env python3

from kaggle.api.kaggle_api_extended import KaggleApi
import json

api = KaggleApi()
api.authenticate()

competition_slug = "openai-to-z-challenge"

try:
    # Get competition details
    competitions = api.competitions_list(search="openai-to-z-challenge")
    competition = None
    for comp in competitions:
        if "openai-to-z-challenge" in comp.ref:
            competition = comp
            break
    
    if competition:
        print(f"Competition: {competition.title}")
        print(f"URL: {competition.ref}")
        print(f"Deadline: {competition.deadline}")
        print(f"Reward: {competition.reward}")
        print()
    
    # Get competition files
    print("Competition files:")
    try:
        files = api.competition_list_files(competition_slug)
        if hasattr(files, 'files'):
            for file in files.files:
                print(f"  - {file.name} ({file.size} bytes)")
        else:
            print(f"  Files object: {files}")
    except Exception as e:
        print(f"  Error listing files: {e}")
    print()
    
    # Try to get competition data
    print("Downloading competition data...")
    try:
        api.competition_download_files(competition_slug, path="./competition_data")
        print("Files downloaded to ./competition_data/")
    except Exception as e:
        print(f"Download failed: {e}")
    
    # Check for kernels with different parameters
    print("Checking for all kernels...")
    kernels = api.kernels_list(competition=competition_slug, sort_by="RELEVANCE")
    if kernels:
        print(f"Found {len(kernels)} kernels:")
        for kernel in kernels:
            print(f"  - {kernel.ref}: {kernel.title}")
            print(f"    Author: {kernel.author}")
            print(f"    Language: {kernel.language}")
            print(f"    Votes: {kernel.totalVotes}")
            print()
    else:
        print("No kernels found")
        
except Exception as e:
    print(f"Error: {e}")