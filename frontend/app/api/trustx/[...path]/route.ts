import type { NextRequest } from "next/server";

/**
 * Server-side proxy to the TrustX API.
 *
 * The browser only talks to this same-origin route. The analyst API key lives in
 * the server-only environment variable TRUSTX_API_KEY and is attached here, so it
 * is never shipped in the client bundle. Only the analyst-level API surface is
 * reachable through this proxy; admin endpoints (model status, audit log) are not.
 *
 * Note: this proxy is not a login. Anyone who can reach this frontend can use the
 * analyst API through it. That is acceptable for a local demo, not for production.
 */
export const dynamic = "force-dynamic";

const BACKEND_URL = (
  process.env.TRUSTX_BACKEND_URL ??
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const ALLOWED_PATHS = [
  /^api\/v1\/transactions(\/|$)/,
  /^api\/v1\/reviews(\/|$)/,
  /^api\/v1\/risk(\/|$)/,
];

function jsonError(status: number, detail: string): Response {
  return Response.json({ detail }, { status });
}

async function proxy(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
): Promise<Response> {
  const apiKey = process.env.TRUSTX_API_KEY;
  if (!apiKey) {
    return jsonError(500, "The frontend server is missing TRUSTX_API_KEY.");
  }

  const { path } = await context.params;
  if (path.some((segment) => segment === ".." || segment === "." || segment === "")) {
    return jsonError(400, "Invalid path.");
  }
  const joined = path.join("/");
  if (!ALLOWED_PATHS.some((pattern) => pattern.test(joined))) {
    return jsonError(403, "This API path is not available through the frontend.");
  }

  const target = `${BACKEND_URL}/${path.map(encodeURIComponent).join("/")}${request.nextUrl.search}`;
  const headers: Record<string, string> = { "X-API-Key": apiKey };
  const contentType = request.headers.get("content-type");
  if (contentType) headers["Content-Type"] = contentType;
  const reviewer = request.headers.get("x-trustx-user");
  if (reviewer) headers["X-TrustX-User"] = reviewer;

  const hasBody = request.method !== "GET" && request.method !== "HEAD";
  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: request.method,
      headers,
      body: hasBody ? await request.text() : undefined,
      cache: "no-store",
      signal: AbortSignal.timeout(30_000),
    });
  } catch {
    return jsonError(502, "The TrustX API is unreachable.");
  }

  const body = await upstream.text();
  return new Response(body, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") ?? "application/json",
      "Cache-Control": "no-store",
    },
  });
}

export { proxy as GET, proxy as POST, proxy as PATCH };
