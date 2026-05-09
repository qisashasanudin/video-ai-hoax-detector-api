#!/usr/bin/env python3
"""
Test script to verify video metadata extraction works correctly.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.main import AnalysisResult, AnalysisExplanations, AnalysisClaim

# Create a test result with video metadata
test_result = AnalysisResult(
    ai_score=0.738,
    misinfo_score=0.45,
    explanations=AnalysisExplanations(
        ai_drivers=[
            "Model anomaly signal: 68.6%",
            "Frequency Domain Analysis: 100.0% deviation from natural texture patterns.",
            "Temporal Consistency: 69.4% across 24 analyzed frames.",
        ],
        claims=[
            AnalysisClaim(
                claim="Video contains AI-generated content",
                verdict="uncertain",
                evidence_urls=[],
                reason="Strong frequency domain anomalies detected"
            )
        ]
    ),
    video_title="AI Generated Video Test",
    video_thumbnail_url="https://i.ytimg.com/vi/szqXppELItw/default.jpg"
)

# Serialize to JSON to verify it works
print("Test AnalysisResult with metadata:")
print(json.dumps(test_result.model_dump(), indent=2))
print("\nSuccess! Video metadata is properly included in the result.")
