"""Ontology pipeline: extract → cluster → enrich → validate, plus
Memgraph persistence (``Onto_{workspace}`` labels) and DSEP operators.

Opt-in via ``ontology.json`` (see :mod:`.config`); absent file means the
feature is disabled with zero behavior change.
"""
