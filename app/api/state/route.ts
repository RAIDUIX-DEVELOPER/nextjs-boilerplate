import { NextRequest, NextResponse } from "next/server";
import { get } from "@vercel/edge-config";

// Expected env vars for write support:
// EDGE_CONFIG_ID (the Edge Config ID)
// EDGE_CONFIG_WRITE_TOKEN (a Vercel token with write access to the Edge Config)
// If write env vars missing, POST becomes a no-op (returns 501).

const EDGE_CONFIG_ID = process.env.EDGE_CONFIG_ID;
const EDGE_CONFIG_WRITE_TOKEN = process.env.EDGE_CONFIG_WRITE_TOKEN;

interface PersistPayload {
  bin_history?: number[];
  bin_records?: any[];
  dz_history?: number[];
  dz_records?: any[];
  dz_state?: any;
  timestamp?: number;
}

export async function GET() {
  try {
    const keys = [
      "bin_history",
      "bin_records",
      "dz_history",
      "dz_records",
      "dz_state",
      "fullstate",
    ];
    const out: Record<string, any> = {};
    for (const k of keys) {
      try {
        out[k] = await get(k);
      } catch {
        /* ignore missing */
      }
    }
    return NextResponse.json({ ok: true, ...out });
  } catch (e: any) {
    return NextResponse.json(
      { ok: false, error: e?.message || "edge-config-get-failed" },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  if (!EDGE_CONFIG_ID || !EDGE_CONFIG_WRITE_TOKEN) {
    // Treat as disabled (200) so client can stop retrying instead of logging 501 spam
    return NextResponse.json({
      ok: false,
      disabled: true,
      error: "write-not-configured",
    });
  }
  let body: PersistPayload | null = null;
  try {
    body = await req.json();
  } catch {
    /* */
  }
  if (!body)
    return NextResponse.json(
      { ok: false, error: "invalid-json" },
      { status: 400 }
    );

  // Prepare patch items only for provided keys
  const items: { op: "upsert"; key: string; value: any }[] = [];
  const map: Record<string, any> = {
    bin_history: body.bin_history,
    bin_records: body.bin_records,
    dz_history: body.dz_history,
    dz_records: body.dz_records,
    dz_state: body.dz_state,
    fullstate: (body as any).fullstate,
  };
  for (const [k, v] of Object.entries(map))
    if (v !== undefined) items.push({ op: "upsert", key: k, value: v });
  if (!items.length)
    return NextResponse.json({ ok: false, error: "no-items" }, { status: 400 });

  try {
    const res = await fetch(
      `https://api.vercel.com/v1/edge-config/${EDGE_CONFIG_ID}/items`,
      {
        method: "PATCH",
        headers: {
          Authorization: `Bearer ${EDGE_CONFIG_WRITE_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ items }),
      }
    );
    if (!res.ok) {
      const txt = await res.text();
      return NextResponse.json(
        { ok: false, error: "edge-config-write-failed", detail: txt },
        { status: 500 }
      );
    }
    return NextResponse.json({ ok: true, updated: items.length });
  } catch (e: any) {
    return NextResponse.json(
      { ok: false, error: e?.message || "edge-config-write-exception" },
      { status: 500 }
    );
  }
}
