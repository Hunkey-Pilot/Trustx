"use client";

import { type FormEvent, useState } from "react";
import type {
  HistoricalTransaction,
  TransactionAnalysisRequest,
  TransactionType,
} from "@/lib/api";

interface TransactionFormProps {
  isLoading: boolean;
  onAnalyze: (payload: TransactionAnalysisRequest) => Promise<void>;
}

const transactionTypes: TransactionType[] = [
  "TRANSFER",
  "CASH_OUT",
  "CASH_IN",
  "PAYMENT",
  "DEBIT",
];

const historyExample = `[
  { "step": 48, "amount": 50, "nameOrig": "C1" }
]`;

function parseHistoricalTransactions(value: string): HistoricalTransaction[] {
  const parsed: unknown = JSON.parse(value);
  if (!Array.isArray(parsed)) {
    throw new Error("Historical transactions must be a JSON array.");
  }

  return parsed.map((entry, index) => {
    if (typeof entry !== "object" || entry === null || Array.isArray(entry)) {
      throw new Error(`Historical row ${index + 1} must be an object.`);
    }

    const row = entry as Record<string, unknown>;
    if (
      typeof row.step !== "number" ||
      !Number.isInteger(row.step) ||
      row.step < 0 ||
      typeof row.amount !== "number" ||
      !Number.isFinite(row.amount) ||
      row.amount < 0 ||
      typeof row.nameOrig !== "string" ||
      row.nameOrig.trim() === ""
    ) {
      throw new Error(
        `Historical row ${index + 1} needs a valid step, amount, and sender ID.`,
      );
    }

    return {
      step: row.step,
      amount: row.amount,
      nameOrig: row.nameOrig,
    };
  });
}

export default function TransactionForm({
  isLoading,
  onAnalyze,
}: TransactionFormProps) {
  const [transactionType, setTransactionType] = useState<TransactionType>("TRANSFER");
  const [amount, setAmount] = useState("");
  const [sender, setSender] = useState("");
  const [recipient, setRecipient] = useState("");
  const [senderOldBalance, setSenderOldBalance] = useState("");
  const [senderNewBalance, setSenderNewBalance] = useState("");
  const [recipientOldBalance, setRecipientOldBalance] = useState("");
  const [recipientNewBalance, setRecipientNewBalance] = useState("");
  const [step, setStep] = useState("");
  const [historyText, setHistoryText] = useState("[]");
  const [formError, setFormError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFormError(null);

    let historicalTransactions: HistoricalTransaction[];
    try {
      historicalTransactions = parseHistoricalTransactions(historyText);
    } catch (error) {
      setFormError(
        error instanceof Error
          ? error.message
          : "Historical transactions must be valid JSON.",
      );
      return;
    }

    await onAnalyze({
      step: Number(step),
      type: transactionType,
      amount: Number(amount),
      nameOrig: sender.trim(),
      oldbalanceOrg: Number(senderOldBalance),
      newbalanceOrig: Number(senderNewBalance),
      nameDest: recipient.trim(),
      oldbalanceDest: Number(recipientOldBalance),
      newbalanceDest: Number(recipientNewBalance),
      historical_transactions: historicalTransactions,
    });
  }

  return (
    <form className="transaction-form" onSubmit={handleSubmit}>
      <div className="form-heading">
        <div>
          <p className="eyebrow">Input</p>
          <h2>Transaction details</h2>
        </div>
        <span className="required-note">All fields required</span>
      </div>

      <div className="form-grid">
        <label className="field">
          <span>Transaction type</span>
          <select
            value={transactionType}
            onChange={(event) => setTransactionType(event.target.value as TransactionType)}
          >
            {transactionTypes.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Step</span>
          <input
            type="number"
            min="0"
            step="1"
            required
            value={step}
            onChange={(event) => setStep(event.target.value)}
            placeholder="49"
          />
        </label>
        <label className="field">
          <span>Amount</span>
          <input
            type="number"
            min="0"
            step="any"
            required
            value={amount}
            onChange={(event) => setAmount(event.target.value)}
            placeholder="100.00"
          />
        </label>
        <label className="field">
          <span>Sender ID</span>
          <input
            type="text"
            required
            value={sender}
            onChange={(event) => setSender(event.target.value)}
            placeholder="C1"
          />
        </label>
        <label className="field">
          <span>Recipient ID</span>
          <input
            type="text"
            required
            value={recipient}
            onChange={(event) => setRecipient(event.target.value)}
            placeholder="M1"
          />
        </label>
      </div>

      <fieldset className="balance-fields">
        <legend>Balances</legend>
        <div className="form-grid balance-grid">
          <label className="field">
            <span>Sender - old balance</span>
            <input
              type="number"
              min="0"
              step="any"
              required
              value={senderOldBalance}
              onChange={(event) => setSenderOldBalance(event.target.value)}
              placeholder="1,000.00"
            />
          </label>
          <label className="field">
            <span>Sender - new balance</span>
            <input
              type="number"
              min="0"
              step="any"
              required
              value={senderNewBalance}
              onChange={(event) => setSenderNewBalance(event.target.value)}
              placeholder="900.00"
            />
          </label>
          <label className="field">
            <span>Recipient - old balance</span>
            <input
              type="number"
              min="0"
              step="any"
              required
              value={recipientOldBalance}
              onChange={(event) => setRecipientOldBalance(event.target.value)}
              placeholder="500.00"
            />
          </label>
          <label className="field">
            <span>Recipient - new balance</span>
            <input
              type="number"
              min="0"
              step="any"
              required
              value={recipientNewBalance}
              onChange={(event) => setRecipientNewBalance(event.target.value)}
              placeholder="600.00"
            />
          </label>
        </div>
      </fieldset>

      <label className="field history-field">
        <span>Historical transactions - optional JSON</span>
        <textarea
          rows={4}
          spellCheck={false}
          value={historyText}
          onChange={(event) => setHistoryText(event.target.value)}
          placeholder={historyExample}
        />
      </label>

      {formError && (
        <p className="form-error" role="alert">
          {formError}
        </p>
      )}

      <button className="analyze-button" type="submit" disabled={isLoading}>
        {isLoading ? "Analyzing transaction..." : "Analyze Transaction"}
        {!isLoading && <span aria-hidden="true">&gt;</span>}
      </button>
    </form>
  );
}