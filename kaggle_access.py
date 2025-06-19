#!/usr/bin/env python3

import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize the API
api = KaggleApi()
api.authenticate()

try:
    # Search for the OpenAI A-to-Z Challenge competition
    competitions = api.competitions_list(search="openai")
    print("OpenAI related competitions:")
    for comp in competitions:
        print(f"- {comp.ref}: {comp.title}")
        if "openai" in comp.ref.lower() or "a-to-z" in comp.title.lower():
            print(f"  Found target competition: {comp.ref}")
            
            # Get competition code
            try:
                kernels = api.kernels_list(competition=comp.ref)
                print(f"\nAvailable kernels for {comp.ref}:")
                for kernel in kernels[:10]:  # Show first 10
                    print(f"  - {kernel.ref}: {kernel.title}")
                    print(f"    Author: {kernel.author}")
                    print(f"    Language: {kernel.language}")
                    print(f"    Score: {kernel.totalVotes}")
                    print()
                break
            except Exception as e:
                print(f"Error accessing kernels: {e}")
                
except Exception as e:
    print(f"Error: {e}")
    # Try direct competition access
    try:
        print("\nTrying direct access to openai-to-z-challenge...")
        kernels = api.kernels_list(competition="openai-to-z-challenge")
        print("Available kernels:")
        for kernel in kernels[:10]:
            print(f"  - {kernel.ref}: {kernel.title}")
            print(f"    Author: {kernel.author}")
            print(f"    Language: {kernel.language}")
            print(f"    Score: {kernel.totalVotes}")
            print()
    except Exception as e2:
        print(f"Direct access also failed: {e2}")