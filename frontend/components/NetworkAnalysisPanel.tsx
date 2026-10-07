import type { NetworkAnalysisResponse } from "@/lib/api";
import { formatNumber } from "@/lib/format";

export default function NetworkAnalysisPanel({
  value,
}: {
  value: NetworkAnalysisResponse | null;
}) {
  const detectedPatterns = value?.patterns.filter((pattern) => pattern.detected) ?? [];

  return (
    <section className="investigation-section">
      <div className="section-title-row">
        <div>
          <p className="eyebrow">Relationship activity</p>
          <h2>Network analysis</h2>
        </div>
        <span className={value?.status === "available" ? "availability available" : "availability"}>
          {value?.status === "available" ? "Available" : "Not available"}
        </span>
      </div>
      {value?.status !== "available" || !value.sender || !value.recipient || !value.relationship ? (
        <p className="muted-copy">
          {value?.error ?? "Network analysis is not available for this transaction."}
        </p>
      ) : (
        <>
          <div className="network-analysis-grid">
            <div className="network-group">
              <h3>Sender activity</h3>
              <dl>
                <div><dt>Outgoing transactions</dt><dd>{value.sender.outgoing_transaction_count}</dd></div>
                <div><dt>Unique recipients</dt><dd>{value.sender.unique_recipient_count}</dd></div>
                <div><dt>Total outgoing amount</dt><dd>{formatNumber(value.sender.total_outgoing_amount)}</dd></div>
              </dl>
            </div>
            <div className="network-group">
              <h3>Recipient activity</h3>
              <dl>
                <div><dt>Incoming transactions</dt><dd>{value.recipient.incoming_transaction_count}</dd></div>
                <div><dt>Unique senders</dt><dd>{value.recipient.unique_sender_count}</dd></div>
                <div><dt>Total incoming amount</dt><dd>{formatNumber(value.recipient.total_incoming_amount)}</dd></div>
              </dl>
            </div>
            <div className="network-group relationship-group">
              <h3>Relationship</h3>
              <dl>
                <div><dt>Previous transactions</dt><dd>{value.relationship.previous_transaction_count}</dd></div>
                <div><dt>Previous amount</dt><dd>{formatNumber(value.relationship.previous_transaction_amount)}</dd></div>
                <div><dt>Relationship status</dt><dd>{value.relationship.relationship_status.replaceAll("_", " ")}</dd></div>
              </dl>
            </div>
          </div>
          <div className="network-patterns">
            <h3>Relationship patterns</h3>
            {detectedPatterns.length ? (
              <ul>
                {detectedPatterns.map((pattern) => (
                  <li key={pattern.pattern}>
                    {pattern.pattern === "HIGH_RECIPIENT_CONNECTIVITY"
                      ? "High recipient connectivity observed (transactions from many distinct senders)"
                      : "High sender connectivity observed (transactions to many distinct recipients)"}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="muted-copy">No unusual connectivity pattern observed</p>
            )}
          </div>
          <p className="source-note">
            {value.source === "legacy_recomputed"
              ? "Legacy record: recomputed from history that was already stored at analysis time (earlier steps only)."
              : "Stored at analysis time from earlier steps only; later transactions do not change this result."}{" "}
            Counts include this transaction. Patterns describe connectivity and are not a determination about any account.
          </p>
        </>
      )}
    </section>
  );
}
