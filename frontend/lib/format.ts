export function formatNumber(value: number | null | undefined, digits = 7): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "Not available";
  return new Intl.NumberFormat("en-US", {
    maximumSignificantDigits: digits,
  }).format(value);
}

export function riskClass(level: string): string {
  return `risk-tag risk-${level.toLowerCase()}`;
}

export function titleCase(value: string): string {
  return value
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/^\w/, (character) => character.toUpperCase());
}

export const MODEL_SCORE_DISCLAIMER =
  "The model score is derived from a PaySim-trained model and is not a calibrated real-world fraud probability.";
