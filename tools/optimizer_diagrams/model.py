"""Data model for the v0.7 optimizer systems maps.

Seven pages share one renderer. Each page is data — panels, legend, boundaries,
receipts, metadata — so a change to the house style lands everywhere at once
and structure tests can assert against the model rather than scraped HTML.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA = "synth.optimizer_diagram.v1"

#: Shared visual language. Every page renders this legend verbatim so a reader
#: who learns it once can read all seven maps.
BASE_LEGEND: tuple[tuple[str, str], ...] = (
    ("┌── ──┐", "runtime / process boundary — one thing that can crash on its own"),
    ("╔══ ══╗", "trust boundary — crossing it changes who may read the data"),
    ("┏━━ ━━┓", "sealed or secret region — never reaches a policy-visible observation"),
    ("──────▶", "control or data flow in the direction of the arrow"),
    ("╌╌╌╌╌╌▶", "advisory, deferred or not-yet-implemented edge"),
    ("[id]", "a durable identifier: run, episode, checkpoint, candidate or digest"),
    ("§", "state that is written to durable storage and survives a restart"),
)


@dataclass(slots=True)
class MetaField:
    label: str
    value: str
    note: str = ""


@dataclass(slots=True)
class Panel:
    """One ASCII map plus the prose that makes it readable."""

    slug: str
    title: str
    caption: str
    ascii_map: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Receipt:
    """A pointer to real evidence. Renders as 'not recorded' when absent."""

    label: str
    identifier: str = ""
    detail: str = ""

    @property
    def present(self) -> bool:
        return bool(self.identifier)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "present": self.present}


@dataclass(slots=True)
class DiagramPage:
    slug: str
    title: str
    subtitle: str
    optimizer: str
    schema_version: str
    verdict: str
    confidence: str
    why: str
    hypothesis: str
    metadata: tuple[MetaField, ...] = ()
    panels: tuple[Panel, ...] = ()
    boundaries: tuple[tuple[str, str], ...] = ()
    receipts: tuple[Receipt, ...] = ()
    extra_legend: tuple[tuple[str, str], ...] = ()
    schema: str = SCHEMA

    @property
    def legend(self) -> tuple[tuple[str, str], ...]:
        return BASE_LEGEND + self.extra_legend

    def required_labels(self) -> set[str]:
        """Every label a structure test should be able to find in the maps."""

        labels: set[str] = set()
        for panel in self.panels:
            for line in panel.ascii_map.splitlines():
                for token in line.split("  "):
                    token = token.strip(" │┌┐└┘╔╗╚╝┏┓┗┛─═━╌╎▶◀▲▼")
                    if len(token) > 3 and token.isupper():
                        labels.add(token)
        return labels

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "slug": self.slug,
            "title": self.title,
            "subtitle": self.subtitle,
            "optimizer": self.optimizer,
            "schema_version": self.schema_version,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "why": self.why,
            "hypothesis": self.hypothesis,
            "metadata": [asdict(field_) for field_ in self.metadata],
            "panels": [panel.to_dict() for panel in self.panels],
            "boundaries": [list(pair) for pair in self.boundaries],
            "receipts": [receipt.to_dict() for receipt in self.receipts],
            "legend": [list(pair) for pair in self.legend],
        }


__all__ = ["BASE_LEGEND", "SCHEMA", "DiagramPage", "MetaField", "Panel", "Receipt"]
