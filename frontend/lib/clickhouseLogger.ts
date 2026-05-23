import { createClient, type ClickHouseClient } from "@clickhouse/client";
import { randomUUID } from "crypto";

let _client: ClickHouseClient | null = null;

function getClient(): ClickHouseClient | null {
  const url = process.env.CLICKHOUSE_URL;
  const username = process.env.CLICKHOUSE_USER;
  const password = process.env.CLICKHOUSE_PASSWORD;
  const database = process.env.CLICKHOUSE_DATABASE;
  if (!url || !username || !password || !database) return null;

  if (_client) return _client;
  _client = createClient({ url, username, password, database });
  return _client;
}

export interface LlmCallLog {
  id?: string;
  input: string;
  output: string;
  model: string;
  latency_ms: number;
}

export function newCallId(): string {
  return `gd_${randomUUID().replace(/-/g, "").slice(0, 12)}`;
}

export async function logLlmCall(entry: LlmCallLog): Promise<string> {
  const id = entry.id ?? newCallId();
  const client = getClient();
  if (!client) return id; // no-op if not configured

  try {
    await client.insert({
      table: "autoval.llm_call_logs",
      values: [
        {
          id,
          input: entry.input,
          output: entry.output,
          model: entry.model,
          latency_ms: entry.latency_ms,
        },
      ],
      format: "JSONEachRow",
    });
  } catch (err) {
    console.error("[clickhouseLogger] insert failed:", err);
  }
  return id;
}
