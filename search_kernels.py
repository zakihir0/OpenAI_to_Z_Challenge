#!/usr/bin/env python3

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

competition_slug = "openai-to-z-challenge"

try:
    # Search for kernels with different parameters
    print("Searching for kernels...")
    
    # Try different sort options
    sort_options = ['hotness', 'relevance', 'voteCount', 'dateCreated']
    
    for sort_by in sort_options:
        print(f"\n--- Sorted by {sort_by} ---")
        try:
            kernels = api.kernels_list(competition=competition_slug, sort_by=sort_by, page_size=10)
            if kernels:
                for i, kernel in enumerate(kernels[:5]):  # Show top 5
                    print(f"{i+1}. {kernel.ref}")
                    print(f"   Title: {kernel.title}")
                    print(f"   Author: {kernel.author}")
                    print(f"   Language: {kernel.language}")
                    print(f"   Votes: {kernel.totalVotes}")
                    print(f"   Medal: {kernel.medal}")
                    print()
            else:
                print("   No kernels found")
        except Exception as e:
            print(f"   Error with {sort_by}: {e}")
    
    # Also try general kernel search
    print("\n--- General kernel search ---")
    try:
        kernels = api.kernels_list(search="openai to z", page_size=10)
        if kernels:
            for kernel in kernels[:5]:
                print(f"- {kernel.ref}: {kernel.title}")
                print(f"  Author: {kernel.author}, Votes: {kernel.totalVotes}")
        else:
            print("No kernels found in general search")
    except Exception as e:
        print(f"General search error: {e}")
        
except Exception as e:
    print(f"Error: {e}")