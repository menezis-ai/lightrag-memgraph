"""
intelligence/ontology/schema.py
===============================
Ontology graph schema: node types, relationship types, properties,
and normative seed data.

Schema is defined in code. Content emerges from the pipeline.
"""

from dataclasses import dataclass, field


# --- Node type definitions ---

@dataclass(frozen=True)
class NodeType:
    label: str
    required_properties: list[str] = field(default_factory=list)
    optional_properties: list[str] = field(default_factory=list)


NODE_TYPES: dict[str, NodeType] = {
    "Term": NodeType(
        label="Term",
        required_properties=["name"],
        optional_properties=["definition", "confidence", "source_doc"],
    ),
    "Role": NodeType(
        label="Role",
        required_properties=["name"],
        optional_properties=["level"],
    ),
    "Team": NodeType(
        label="Team",
        required_properties=["name"],
        optional_properties=["org_unit"],
    ),
    "Tool": NodeType(
        label="Tool",
        required_properties=["name"],
        optional_properties=["version", "vendor"],
    ),
    "Process": NodeType(
        label="Process",
        required_properties=["name"],
        optional_properties=["type"],
    ),
    "Domain": NodeType(
        label="Domain",
        required_properties=["name"],
        optional_properties=["description"],
    ),
    "Document": NodeType(
        label="Document",
        required_properties=["doc_id"],
        optional_properties=["title", "path"],
    ),
    "Methodology": NodeType(
        label="Methodology",
        required_properties=["name"],
        optional_properties=["version", "framework"],
    ),
    "Environment": NodeType(
        label="Environment",
        required_properties=["name"],
        optional_properties=["tier"],
    ),
    "SLA": NodeType(
        label="SLA",
        required_properties=["priority"],
        optional_properties=["gtr_hours", "description"],
    ),
    "Asset": NodeType(
        label="Asset",
        required_properties=["name"],
        optional_properties=["type", "criticality"],
    ),
}

RELATION_TYPES: list[str] = [
    "SYNONYM",
    "RELATED_TO",
    "CAUSED_BY",
    "MITIGATED_BY",
    "DIAGNOSED_WITH",
    "CO_OCCURS",
    "OWNS",
    "USES",
    "FOLLOWS",
    "ESCALATES_TO",
    "DEPENDS_ON",
    "DOCUMENTED_IN",
    "PART_OF",
    "REPLACES",
    "REQUIRES_APPROVAL",
    "TRIGGERS",
]

RELATION_PROPERTIES: list[str] = ["confidence", "source_doc", "created_at"]


# --- Normative seed data (pre-loaded on initialize()) ---

SEED_METHODOLOGIES: list[dict[str, str]] = [
    {"name": "ITIL", "version": "v4", "framework": "Service Management"},
    {"name": "SAFe", "version": "6.0", "framework": "Scaled Agile"},
    {"name": "DevOps", "version": "", "framework": "Culture & Practices"},
    {"name": "SRE", "version": "", "framework": "Site Reliability Engineering"},
    {"name": "COBIT", "version": "2019", "framework": "IT Governance"},
    {"name": "ISO 27001", "version": "2022", "framework": "Information Security"},
    {"name": "DORA", "version": "", "framework": "Digital Operational Resilience"},
    {"name": "Lean Six Sigma", "version": "", "framework": "Process Improvement"},
    {"name": "TOGAF", "version": "10", "framework": "Enterprise Architecture"},
]

SEED_SLAS: list[dict[str, str | int]] = [
    {"priority": "P1", "gtr_hours": 1, "description": "Critical - Service down"},
    {"priority": "P2", "gtr_hours": 4, "description": "Major - Degraded service"},
    {"priority": "P3", "gtr_hours": 8, "description": "Minor - Workaround exists"},
    {"priority": "P4", "gtr_hours": 48, "description": "Low - Informational"},
]

SEED_ENVIRONMENTS: list[dict[str, str]] = [
    {"name": "Production", "tier": "prod"},
    {"name": "Pre-production", "tier": "preprod"},
    {"name": "UAT", "tier": "uat"},
    {"name": "Development", "tier": "dev"},
    {"name": "Disaster Recovery", "tier": "dr"},
]
