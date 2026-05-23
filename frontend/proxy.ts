import { NextRequest, NextResponse } from "next/server";

// Demo: land users directly on /chat (skipped onboarding for QR-code audience flow).
// The demo profile is pre-baked in user-profile.json so Guardia has health context
// (warfarin, metformin, peanut/tree nut allergies) without an onboarding step.
export async function proxy(req: NextRequest) {
  const { pathname } = req.nextUrl;
  if (pathname !== "/") return NextResponse.next();
  return NextResponse.redirect(new URL("/chat", req.url));
}

export const config = {
  matcher: ["/"],
};
