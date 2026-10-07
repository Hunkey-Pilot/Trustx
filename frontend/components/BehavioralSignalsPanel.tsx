import type { BehavioralSignalsResponse } from "@/lib/api";
import { formatNumber } from "@/lib/format";

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

export default function BehavioralSignalsPanel({
  value,
}: {
  value: BehavioralSignalsResponse | null;
}) {
  const available = value?.status === "available" && value.sender && value.recipient;
  return (
    <section className="investigation-section">
      <div className="section-title-row">
        <div>
          <p className="eyebrow">Sender and recipient history</p>
          <h2>Behavioral signals</h2>
        </div>
        <span className={available ? "availability available" : "availability"}>
          {available ? "Available" : "Not available"}
        </span>
      </div>
      {available && value?.sender && value.recipient && value.sender_recipient ? (
        <>
          <div className="network-analysis-grid">
            <div className="network-group">
              <h3>Sender activity</h3>
              <dl>
                <Row label="Prior transactions (5 min)" value={String(value.sender.transaction_count_5m)} />
                <Row label="Prior transactions (1 h)" value={String(value.sender.transaction_count_1h)} />
                <Row label="Prior transactions (24 h)" value={String(value.sender.transaction_count_24h)} />
                <Row label="Amount sent (24 h)" value={formatNumber(value.sender.amount_sum_24h)} />
                <Row label="Prior average amount" value={formatNumber(value.sender.historical_average_amount)} />
                <Row
                  label="Amount vs prior average"
                  value={
                    value.sender.amount_to_historical_average === null
                      ? "No history"
                      : `${formatNumber(value.sender.amount_to_historical_average, 4)}x`
                  }
                />
                <Row label="Prior maximum amount" value={formatNumber(value.sender.historical_max_amount)} />
              </dl>
            </div>
            <div className="network-group">
              <h3>Recipient activity</h3>
              <dl>
                <Row label="Prior transactions received" value={String(value.recipient.transaction_count)} />
                <Row label="Distinct prior senders" value={String(value.recipient.unique_senders)} />
                <Row label="Distinct senders (1 h)" value={String(value.recipient.unique_senders_1h)} />
                <Row label="Distinct senders (24 h)" value={String(value.recipient.unique_senders_24h)} />
                <Row label="Total received" value={formatNumber(value.recipient.amount_sum)} />
              </dl>
            </div>
            <div className="network-group relationship-group">
              <h3>Sender → recipient</h3>
              <dl>
                <Row
                  label="Recipient"
                  value={
                    value.sender_recipient.is_new_recipient_for_sender
                      ? "First transaction to this recipient"
                      : "Seen before"
                  }
                />
                <Row label="Prior transactions" value={String(value.sender_recipient.pair_transaction_count)} />
                <Row label="Prior amount" value={formatNumber(value.sender_recipient.pair_amount_sum)} />
                <Row
                  label="Minutes since previous"
                  value={
                    value.sender_recipient.minutes_since_previous_pair_transaction === null
                      ? "n/a"
                      : formatNumber(value.sender_recipient.minutes_since_previous_pair_transaction, 4)
                  }
                />
              </dl>
            </div>
          </div>
          <p className="source-note">
            Calculated from analyses stored before this transaction (earlier steps only).
            {value.source === "legacy_recomputed" &&
              " Legacy record: recomputed from history that was already stored at analysis time."}
          </p>
        </>
      ) : (
        <p className="muted-copy">
          {value?.error ?? "Behavioral signals are not available for this transaction."}
        </p>
      )}
    </section>
  );
}
