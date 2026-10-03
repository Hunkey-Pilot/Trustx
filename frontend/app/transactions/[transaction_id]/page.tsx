import TransactionInvestigation from "@/components/TransactionInvestigation";

interface TransactionPageProps {
  params: Promise<{ transaction_id: string }>;
}

export default async function TransactionPage({ params }: TransactionPageProps) {
  const { transaction_id: transactionId } = await params;
  return <TransactionInvestigation key={transactionId} transactionId={transactionId} />;
}