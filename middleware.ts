import { NextResponse } from "next/server";
import { get } from "@vercel/edge-config";

export const config = { matcher: ["/welcome"] };

export async function middleware() {
  try {
    const greeting = await get("greeting");
    return NextResponse.json({ greeting: greeting ?? "no-greeting-set" });
  } catch (e: any) {
    return NextResponse.json(
      { error: e?.message || "edge-config-error" },
      { status: 500 }
    );
  }
}
