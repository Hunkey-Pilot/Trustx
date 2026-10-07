import Link from "next/link";

interface SiteHeaderProps {
  current?: "dashboard" | "transactions" | "analysis";
}

export default function SiteHeader({ current = "dashboard" }: SiteHeaderProps) {
  return (
    <header className="topbar">
      <Link className="brand" href="/" aria-label="TrustX dashboard">
        <span className="brand-mark">TX</span>
        <span>TrustX</span>
      </Link>
      <nav className="topbar-nav" aria-label="Main navigation">
        <Link
          className={current === "dashboard" ? "nav-link active" : "nav-link"}
          href="/"
        >
          Dashboard
        </Link>
        <Link className="nav-link" href="/#review-queue">
          Review queue
        </Link>
        <Link
          className={current === "transactions" ? "nav-link active" : "nav-link"}
          href="/#transactions"
        >
          Transactions
        </Link>
        <Link
          className={current === "analysis" ? "nav-link active" : "nav-link"}
          href="/analyze"
        >
          Analyze
        </Link>
      </nav>
      <div className="topbar-meta">
        <span className="status-dot" aria-hidden="true" />
        <span>Risk intelligence</span>
      </div>
    </header>
  );
}