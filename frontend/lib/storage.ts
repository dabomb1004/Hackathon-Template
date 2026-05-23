import fs from "fs";
import path from "path";

const LOCAL_PATH = path.join(process.cwd(), "user-profile.json");

function readProfile(): Record<string, unknown> {
  try {
    if (fs.existsSync(LOCAL_PATH)) {
      return JSON.parse(fs.readFileSync(LOCAL_PATH, "utf-8"));
    }
  } catch {}
  return {};
}

function writeProfile(data: Record<string, unknown>): void {
  // On Vercel serverless /var/task is read-only — silently skip writes there.
  // For the demo the profile is pre-baked at deploy time, so in-conversation
  // memory updates just don't persist between requests. Local dev still works.
  try {
    fs.writeFileSync(LOCAL_PATH, JSON.stringify(data, null, 2));
  } catch (err) {
    if ((err as NodeJS.ErrnoException)?.code !== "EROFS") {
      console.warn("[storage] writeProfile failed (non-EROFS):", err);
    }
  }
}

export async function getUser(): Promise<Record<string, unknown>> {
  return readProfile();
}

export async function saveUser(user: Record<string, unknown>): Promise<void> {
  writeProfile(user);
}

export async function saveUserMemory(key: string, value: unknown): Promise<void> {
  const profile = readProfile();
  profile[key] = value;
  writeProfile(profile);
}

export interface UserProfile {
  id: string;
  name?: string;
  age?: number;
  height?: string;
  weight?: string;
  blood_type?: string;
  medical_history?: string[];
  current_medications?: string[];
  insurance?: {
    provider?: string;
    plan?: string;
    member_id?: string;
    group_number?: string;
    copay?: string;
    deductible?: string;
  };
  onboarded?: boolean;
  [key: string]: unknown;
}
