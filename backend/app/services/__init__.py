"""Deterministic pipeline services.

The split the PRD insists on: everything here is conventional software. Only
`ai_highlights` talks to a model, and it makes semantic judgements only -- it
never controls a media operation.
"""
