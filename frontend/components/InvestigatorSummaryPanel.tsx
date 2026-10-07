import type { InvestigatorSummary } from "@/lib/api";

export default function InvestigatorSummaryPanel({
  value,
}: {
  value: InvestigatorSummary | null | undefined;
}) {
  return (
    <section className="investigation-section">
      <div className="section-title-row">
        <div>
          <p className="eyebrow">Generated from stored facts</p>
          <h2>Investigator summary</h2>
        </div>
        <span className="availability">{value ? "Deterministic" : "Not available"}</span>
      </div>
      {value ? (
        <>
          <p className="summary-text">{value.investigator_summary}</p>
          {value.key_facts.length > 0 && (
            <ul className="facts-list">
              {value.key_facts.map((fact) => (
                <li key={fact}>{fact}</li>
              ))}
            </ul>
          )}
          <div className="recommendation-block">
            <h3>Investigation recommendations</h3>
            {value.investigation_recommendations.length ? (
              <ul className="recommendation-list">
                {value.investigation_recommendations.map((action) => (
                  <li key={action.code}>
                    <strong>{action.label}</strong>
                    <span>{action.rationale}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="muted-copy">No specific investigation action suggested.</p>
            )}
          </div>
          <p className="source-note">{value.disclaimer}</p>
        </>
      ) : (
        <p className="muted-copy">Not available</p>
      )}
    </section>
  );
}
