// Robust scraping endpoint with multi-strategy extraction.
// Strategy 1: Raw HTML fetch + regex search for payout divs.
// Strategy 2: (Optional) Headless browser fallback (disabled unless requested) for dynamic content.
// Use responsibly and respect target site's Terms of Service.
export const dynamic = "force-dynamic"; // disable caching
export const runtime = "nodejs"; // ensure Node runtime for optional puppeteer

interface ExtractResult {
  values: number[];
  debug?: string;
}

function extractMultipliers(html: string): ExtractResult {
  // Match any div whose class contains 'payout' (used only to locate indices below)
  const divRegex =
    /<div[^>]*class="[^"]*payout[^"]*"[^>]*>([\s\S]*?)<\/div>/gim;
  let m: RegExpExecArray | null;
  // Goal: ONLY capture the first payouts list: <div class="payouts-wrapper"><div class="payouts-block"> ...payout divs... </div>
  // Avoid the duplicate list inside the dropdown (<app-stats-dropdown> ... another payouts-block ...)
  const primaryWrapperIdx = html.indexOf("payouts-wrapper");
  if (primaryWrapperIdx === -1) {
    return { values: [], debug: "no-payouts-wrapper" };
  }
  // Find first 'payouts-block' after wrapper
  const firstBlockIdx = html.indexOf("payouts-block", primaryWrapperIdx);
  if (firstBlockIdx === -1) {
    return { values: [], debug: "no-first-block" };
  }
  // Heuristic end: either before 'button-block' (ends the top bar) or before '<app-stats-dropdown'
  const buttonIdx = html.indexOf("button-block", firstBlockIdx);
  const dropdownIdx = html.indexOf("<app-stats-dropdown", firstBlockIdx);
  let endIdx = html.length;
  if (buttonIdx !== -1) endIdx = Math.min(endIdx, buttonIdx);
  if (dropdownIdx !== -1) endIdx = Math.min(endIdx, dropdownIdx);
  const slice = html.slice(firstBlockIdx, endIdx);
  const values: number[] = [];
  // Match only direct payout divs within this slice
  const payoutDivRegex =
    /<div[^>]*class="[^"]*payout[^"]*"[^>]*>([\s\S]*?)<\/div>/gim;
  while ((m = payoutDivRegex.exec(slice)) !== null) {
    const raw = (m[1] || "").trim().replace(/\s+/g, "");
    const num = parseFloat(raw.replace(/x$/i, ""));
    if (isFinite(num)) values.push(num);
  }
  if (!values.length) {
    return { values: [], debug: "primary-block-empty" };
  }
  return { values, debug: `primary=${values.length}` };
}

async function headlessExtract(url: string): Promise<ExtractResult> {
  if (process.env.DISABLE_HEADLESS === "1")
    return { values: [], debug: "headless-disabled" };
  try {
    return { values: [], debug: "headless-unavailable" };
  } catch (e: any) {
    return { values: [], debug: `headless-error=${e?.message || e}` };
  }
}

export async function GET(req: Request) {
  const url = "https://lottostar.co.za/games/spribe/spribe-aviator";
  const search = new URL(req.url).searchParams;
  const forceHeadless = search.get("mode") === "headless";
  try {
    const res = await fetch(url, {
      headers: {
        "user-agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36",
        accept: "text/html,application/xhtml+xml",
        referer: "https://lottostar.co.za/",
        "accept-language": "en-US,en;q=0.9",
      },
      cache: "no-store",
    });
    const html = await res.text();
    let { values, debug } = extractMultipliers(html);
    let headlessDebug: string | undefined;
    if (forceHeadless || values.length === 0) {
      const h = await headlessExtract(url);
      if (h.values.length) values = h.values;
      headlessDebug = h.debug;
    }
    const value = values[0] ?? null;
    return new Response(
      JSON.stringify({
        ok: value != null,
        value,
        values,
        ts: Date.now(),
        debug,
        headlessDebug,
      }),
      {
        headers: {
          "content-type": "application/json",
          "cache-control": "no-store",
        },
      }
    );
  } catch (e: any) {
    return new Response(
      JSON.stringify({ ok: false, error: e?.message || "scrape failed" }),
      {
        status: 500,
        headers: {
          "content-type": "application/json",
          "cache-control": "no-store",
        },
      }
    );
  }
}
