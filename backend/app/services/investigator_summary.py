"""Deterministic investigator summary built only from stored analysis facts.

No language model is involved: every sentence is produced from a field that
exists in the stored analysis. The text is decision support, never a verdict.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.schemas.transactions import InvestigatorSummary
from app.services.recommendations import (
    SENDER_AMOUNT_MULTIPLE_NOTABLE,
    SENDER_VELOCITY_1H_NOTABLE,
    build_recommendations,
)


SUMMARY_DISCLAIMER = (
    "This summary is generated from model outputs and stored signals. The model "
    "score comes from a PaySim-trained model and is not a calibrated real-world "
    "fraud probability. It is decision support and does not establish that the "
    "transaction is fraudulent."
)

_CLOSING = {
    "LOW": "The model did not flag elevated risk; routine monitoring applies.",
    "MEDIUM": "Additional verification and monitoring are suggested.",
    "HIGH": "Analyst review is recommended before any decision is taken.",
    "CRITICAL": "Analyst review is recommended; the recommended action is a temporary hold pending that review.",
}


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _get(mapping: Any, *path: str) -> Any:
    current = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _shap_sentence(explanation: Mapping[str, Any] | None) -> tuple[str | None, list[str]]:
    if not isinstance(explanation, Mapping) or explanation.get("status") != "available":
        return None, []
    features = explanation.get("top_features") or []
    parts: list[str] = []
    facts: list[str] = []
    for item in features[:3]:
        name = item.get("feature")
        direction = item.get("direction")
        value = _num(item.get("value"))
        if not name:
            continue
        effect = {
            "increases_fraud_prediction": "raised the score",
            "decreases_fraud_prediction": "lowered the score",
        }.get(direction, "had a neutral effect")
        value_text = f" = {value:,.4g}" if value is not None else ""
        parts.append(f"{name}{value_text} ({effect})")
        facts.append(f"SHAP: {name}{value_text} {effect}")
    if not parts:
        return None, facts
    return "The main model attributions were " + "; ".join(parts) + ".", facts


def _behavior_observations(behavioral: Mapping[str, Any] | None) -> tuple[list[str], list[str]]:
    sentences: list[str] = []
    facts: list[str] = []
    if not isinstance(behavioral, Mapping) or behavioral.get("status") != "available":
        return sentences, facts
    sender = behavioral.get("sender") or {}
    recipient = behavioral.get("recipient") or {}
    pair = behavioral.get("sender_recipient") or {}

    average = _num(sender.get("historical_average_amount"))
    multiple = _num(sender.get("amount_to_historical_average"))
    if average is None:
        sentences.append("No earlier analyzed transactions from this sender were available for comparison.")
        facts.append("Sender history: none available")
    elif multiple is not None:
        facts.append(f"Amount is {multiple:.2f}x the sender's prior average")
        if multiple >= SENDER_AMOUNT_MULTIPLE_NOTABLE:
            sentences.append(
                f"The amount is {multiple:.1f} times the sender's previously observed average."
            )
    count_1h = _num(sender.get("transaction_count_1h"))
    if count_1h is not None:
        facts.append(f"Sender prior transactions in last hour: {int(count_1h)}")
        if count_1h >= SENDER_VELOCITY_1H_NOTABLE:
            sentences.append(
                f"The sender has {int(count_1h)} earlier analyzed transactions in the last hour."
            )
    if pair.get("is_new_recipient_for_sender") is True:
        facts.append("First analyzed transaction between this sender and recipient")
        sentences.append("This is the first analyzed transaction between this sender and recipient.")
    unique_24h = _num(recipient.get("unique_senders_24h"))
    if unique_24h is not None and unique_24h >= 3:
        facts.append(f"Recipient distinct senders (24h): {int(unique_24h)}")
        sentences.append(
            f"The recipient received transactions from {int(unique_24h)} distinct senders in the last 24 hours."
        )
    return sentences, facts


def _network_observations(network: Mapping[str, Any] | None) -> tuple[list[str], list[str]]:
    sentences: list[str] = []
    facts: list[str] = []
    if not isinstance(network, Mapping) or network.get("status") != "available":
        return sentences, facts
    relationship = _get(network, "relationship", "relationship_status")
    if relationship:
        facts.append(f"Relationship: {str(relationship).replace('_', ' ').lower()}")
    recipient_senders = _num(_get(network, "recipient", "unique_sender_count"))
    sender_recipients = _num(_get(network, "sender", "unique_recipient_count"))
    if recipient_senders is not None:
        facts.append(f"Recipient unique senders (incl. this): {int(recipient_senders)}")
    if sender_recipients is not None:
        facts.append(f"Sender unique recipients (incl. this): {int(sender_recipients)}")
    for pattern in network.get("patterns") or []:
        if not pattern.get("detected"):
            continue
        if pattern.get("pattern") == "HIGH_RECIPIENT_CONNECTIVITY":
            sentences.append("The recipient shows high connectivity (transactions from many distinct senders).")
        elif pattern.get("pattern") == "HIGH_SENDER_CONNECTIVITY":
            sentences.append("The sender shows high connectivity (transactions to many distinct recipients).")
    return sentences, facts


def build_investigator_summary(
    analysis: Mapping[str, Any],
    amount_sensitive: bool = False,
) -> InvestigatorSummary:
    """Build the summary from a mapping with the stored analysis fields."""
    level = str(analysis.get("risk_level", "LOW"))
    transaction_type = str(analysis.get("type", "transaction")).replace("_", " ").lower()
    probability = _num(analysis.get("fraud_probability"))
    risk_score = _num(analysis.get("risk_score"))
    anomaly = _num(analysis.get("anomaly_signal"))
    amount = _num(analysis.get("amount"))

    score_text = f"model score {probability:.4f}, uncalibrated" if probability is not None else "model score unavailable"
    sentences = [
        f"This {transaction_type} transaction received a {level} model risk assessment ({score_text})."
    ]
    facts = [
        f"Risk level: {level}",
        f"Recommended action: {analysis.get('recommended_action', 'n/a')}",
    ]
    if probability is not None:
        facts.append(f"Model score (uncalibrated): {probability:.6f}")
    if risk_score is not None:
        facts.append(f"Risk score: {risk_score:.2f}")
    if amount is not None:
        facts.append(f"Amount: {amount:,.2f}")

    shap_sentence, shap_facts = _shap_sentence(analysis.get("explanation"))
    if shap_sentence:
        sentences.append(shap_sentence)
    facts.extend(shap_facts)

    behavior_sentences, behavior_facts = _behavior_observations(analysis.get("behavioral_signals"))
    sentences.extend(behavior_sentences)
    facts.extend(behavior_facts)

    network_sentences, network_facts = _network_observations(analysis.get("network_signals"))
    sentences.extend(network_sentences)
    facts.extend(network_facts)

    if anomaly is not None:
        facts.append(f"Isolation Forest anomaly value: {anomaly:.4f}")
        if anomaly < 0:
            sentences.append(
                "The Isolation Forest anomaly value is below zero, meaning the transaction looks "
                "unusual compared with the transactions it was trained on; this value is shown "
                "for context and is not part of the risk score."
            )

    sentences.append(_CLOSING.get(level, _CLOSING["LOW"]))

    recommendations = build_recommendations(
        level,
        behavioral_signals=analysis.get("behavioral_signals"),
        network_signals=analysis.get("network_signals"),
        amount_sensitive=amount_sensitive,
    )
    return InvestigatorSummary(
        investigator_summary=" ".join(sentences),
        key_facts=facts,
        investigation_recommendations=recommendations,
        disclaimer=SUMMARY_DISCLAIMER,
    )
