"use client";

import { type FormEvent, useEffect, useState } from "react";
import {
  createReview,
  getReview,
  updateReview,
  type ReviewResponse,
  type ReviewStatus,
  type RiskLevel,
} from "@/lib/api";
import { titleCase } from "@/lib/format";

const STATUSES: ReviewStatus[] = [
  "OPEN",
  "UNDER_REVIEW",
  "DISMISSED",
  "ESCALATED",
  "CONFIRMED_SUSPICIOUS",
];

export default function AnalystReview({
  transactionId,
  riskLevel,
  onChanged,
}: {
  transactionId: string;
  riskLevel: RiskLevel;
  onChanged?: (review: ReviewResponse) => void;
}) {
  const [review, setReview] = useState<ReviewResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [status, setStatus] = useState<ReviewStatus>("UNDER_REVIEW");
  const [note, setNote] = useState("");
  const [decision, setDecision] = useState("");
  const [reviewer, setReviewer] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const existing = await getReview(transactionId);
        if (cancelled) return;
        setReview(existing);
        if (existing) {
          setStatus(existing.status);
          setNote(existing.analyst_note ?? "");
          setDecision(existing.decision ?? "");
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Unable to load the review.");
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [transactionId]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSaving(true);
    setError(null);
    setSaved(false);
    const input = {
      status,
      analyst_note: note.trim() === "" ? null : note.trim(),
      decision: decision.trim() === "" ? null : decision.trim(),
    };
    try {
      const result = review
        ? await updateReview(transactionId, input, reviewer)
        : await createReview(transactionId, input, reviewer);
      setReview(result);
      setSaved(true);
      onChanged?.(result);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Unable to save the review.");
    } finally {
      setIsSaving(false);
    }
  }

  const needsReview = riskLevel === "HIGH" || riskLevel === "CRITICAL";

  return (
    <section className="investigation-section review-section">
      <div className="section-title-row">
        <div>
          <p className="eyebrow">Human review</p>
          <h2>Analyst review</h2>
        </div>
        <span className={review ? `status-pill status-${review.status.toLowerCase()}` : "availability"}>
          {review ? titleCase(review.status) : "No review yet"}
        </span>
      </div>
      <p className="source-note">
        {needsReview
          ? `${riskLevel} risk assessments should be reviewed by a human analyst. `
          : ""}
        The analyst decision is recorded separately; it never changes the model score, risk level or recommended action.
      </p>
      {isLoading ? (
        <p className="muted-copy" role="status">Loading review...</p>
      ) : (
        <form className="review-form" onSubmit={handleSubmit}>
          <div className="review-fields">
            <label className="field">
              <span>Status</span>
              <select value={status} onChange={(event) => setStatus(event.target.value as ReviewStatus)}>
                {STATUSES.map((value) => (
                  <option key={value} value={value}>{titleCase(value)}</option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Decision (optional)</span>
              <input
                type="text"
                maxLength={200}
                value={decision}
                onChange={(event) => setDecision(event.target.value)}
                placeholder="e.g. Customer contacted, payment confirmed"
              />
            </label>
            <label className="field">
              <span>Reviewer name (optional)</span>
              <input
                type="text"
                maxLength={48}
                value={reviewer}
                onChange={(event) => setReviewer(event.target.value)}
                placeholder="Shown in the audit trail"
              />
            </label>
          </div>
          <label className="field">
            <span>Analyst note</span>
            <textarea
              rows={3}
              maxLength={2000}
              value={note}
              onChange={(event) => setNote(event.target.value)}
              placeholder="Observations, contact attempts, rationale"
            />
          </label>
          <div className="review-actions">
            <button type="submit" className="analyze-button review-submit" disabled={isSaving}>
              {isSaving ? "Saving..." : review ? "Update review" : "Save review"}
            </button>
            {saved && <span className="save-confirmation" role="status">Review saved.</span>}
          </div>
          {error && (
            <p className="form-error" role="alert">{error}</p>
          )}
          {review && (
            <dl className="review-meta">
              <div><dt>Reviewed by</dt><dd>{review.reviewed_by ?? "n/a"}</dd></div>
              <div><dt>Reviewed at</dt><dd>{review.reviewed_at ? new Date(review.reviewed_at).toLocaleString() : "n/a"}</dd></div>
              <div><dt>Created</dt><dd>{new Date(review.created_at).toLocaleString()}</dd></div>
              <div><dt>Last updated</dt><dd>{new Date(review.updated_at).toLocaleString()}</dd></div>
            </dl>
          )}
        </form>
      )}
    </section>
  );
}
