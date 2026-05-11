import subprocess
import json

prompt = """
You are a search query generator. Based on the following video context, generate short Google Search keywords to verify the claims in the video.
Create exactly 2 versions of the search query: 1 in English and 1 in Bahasa Indonesia.
Return ONLY a valid JSON array of strings containing the queries.
Example: ["Ali Khamenei killed Tehran", "Ali Khamenei tewas di Teheran"]

Video Context:
Title: BREAKING NEWS: Ayatollah Ali Khamenei Killed in US Airstrike in Tehran
Description: Reports indicate that Iran's supreme leader was assassinated in a drone strike.
Transcript: None
"""

try:
    completed = subprocess.run(
        ["ollama", "run", "gemma4", "--format", "json", "--nowordwrap", prompt],
        capture_output=True,
        text=True,
        check=True,
    )
    print("OLLAMA RESPONSE:", completed.stdout.strip())
except Exception as e:
    print("ERROR:", e)
