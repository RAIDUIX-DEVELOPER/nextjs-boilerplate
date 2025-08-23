"use client";
import { useEffect, useState } from "react";

interface Options {
  intervalMs?: number;
}

export function useAutoScrape(
  onValue: (v: number) => void,
  deps: any[] = [],
  opts: Options = {}
) {
  const { intervalMs = 5000 } = opts;
  const [enabled, setEnabled] = useState(false);
  // Each poll may return multiple values (values[]); we take the first for training consistency.
  const [status, setStatus] = useState<string>("idle");

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let timer: any;
    const tick = async () => {
      try {
        setStatus("fetching");
        const r = await fetch("/api/scrape?" + Date.now(), {
          cache: "no-store",
        });
        if (!r.ok) throw new Error(r.status + "");
        const data = await r.json();
        if (!cancelled && data) {
          const arr: number[] = Array.isArray(data.values)
            ? data.values
            : typeof data.value === "number"
            ? [data.value]
            : [];
          if (arr.length) {
            const first = arr[0];
            if (isFinite(first)) {
              onValue(first);
              if (process.env.NODE_ENV !== "production") {
                console.debug("[auto-scrape] ingested", first, {
                  count: arr.length,
                  debug: data.debug,
                  headless: data.headlessDebug,
                });
              }
            }
          } else if (process.env.NODE_ENV !== "production") {
            console.debug("[auto-scrape] no values", data);
          }
        }
        setStatus("ok");
      } catch (e: any) {
        if (!cancelled) setStatus("err");
      } finally {
        if (!cancelled) timer = setTimeout(tick, intervalMs);
      }
    };
    tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [enabled, intervalMs, onValue, ...deps]);

  return { enabled, setEnabled, status };
}
