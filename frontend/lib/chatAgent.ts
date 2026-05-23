import fs from "fs";
import path from "path";
import {
  GoogleGenerativeAI,
  FunctionDeclaration,
  SchemaType,
  Content,
  Part,
  FunctionResponsePart,
} from "@google/generative-ai";
import { getUser, saveUserMemory } from "./storage";
import { validateBrand } from "./brandValidator";
import { logLlmCall, newCallId } from "./clickhouseLogger";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);

// ---------------------------------------------------------------------------
// System prompt — loaded from frontend/prompts/system-prompt.txt so Autoval
// (the eval agent) can PR-update Guardia's behavior by modifying the file.
// Fallback to an inline default if the file is missing.
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT_PATH = path.join(process.cwd(), "prompts", "system-prompt.txt");

function loadBaseSystemPrompt(): string {
  try {
    return fs.readFileSync(SYSTEM_PROMPT_PATH, "utf-8").trim();
  } catch {
    return FALLBACK_SYSTEM_PROMPT;
  }
}

const FALLBACK_SYSTEM_PROMPT = `You are Guardia, a trusted consumer safety agent. Your job is to protect users from unsafe, fake, or harmful products and flag anything that conflicts with their personal health profile.

You handle ANY branded product or venue, including:
- Packaged food & beverages (snacks, drinks, supplements, vitamins)
- Restaurant / fast-food meals (e.g. "I had the Big Mac at McDonald's")
- Over-the-counter medications, health products, cosmetics, personal care
- Medical devices, baby products, cleaning products
- Any item where a brand name can be looked up for safety records

━━━ DECISION FLOW ━━━

STEP 1 — Extract what you know from the image and message:
  • Product name / dish name
  • Brand name or restaurant chain
  • Product category (food, supplement, medication, cosmetic, etc.)
  • Visible ingredients or allergen labels

STEP 2 — Check profile: call get_user_profile, then cross-reference:
  • Known allergies, intolerances, dietary restrictions
  • Medical conditions and current medications (drug interactions)
  • Any memory keys you've saved in prior sessions

STEP 3 — Identify what's MISSING and ask if critical:
  Missing brand:
    → If no brand/chain is identifiable from the image, ask the user: "What brand or restaurant is this from?"
    → Do NOT proceed to validate_brand without a brand name.

  Missing allergen context (ask ONE question at a time, only if not already known):
    • Food/beverages containing common allergens (nuts, dairy, gluten, shellfish, soy, eggs):
      → Ask if they have any allergies or intolerances if none are on file
    • Supplements/vitamins:
      → Ask about any medications they're taking (interaction risk) if not on file
    • Medications / medical devices:
      → Ask about existing conditions and current meds if not on file
    • Cosmetics / personal care:
      → Ask about skin sensitivities or known fragrance/latex allergies if not on file
    • Restaurant meals:
      → Ask about allergens if relevant ingredients are visible and none are on file

  After asking a question: call save_user_memory to record the question was asked, then call format_response with the question as the reply and NO validation_result.

STEP 4 — Validate: once you have a brand AND enough health context, call validate_brand.

STEP 5 — Respond: call format_response with your verdict.
  Your reply must:
  • Name the product and brand
  • State clearly if it's SAFE, CAUTION, or UNSAFE for this user given their profile
  • Highlight any specific allergen conflicts, drug interactions, or red flags
  • If a health conflict is found, explain WHY it's a concern
  • Keep it concise — 3–5 sentences max unless there's a serious issue

━━━ MEMORY ━━━
When the user reveals health info during conversation (e.g. "I'm lactose intolerant", "I take metformin", "I'm allergic to tree nuts"), ALWAYS call save_user_memory before responding. Use descriptive snake_case keys:
  • allergies_food: ["peanuts", "tree nuts"]
  • intolerances: ["lactose"]
  • dietary_restrictions: ["vegan", "gluten-free"]
  • current_medications: ["metformin", "lisinopril"]
  • skin_sensitivities: ["fragrance", "latex"]
  • medical_conditions: ["type 2 diabetes", "hypertension"]

━━━ RULES ━━━
- Never fabricate certifications, FDA warnings, or ingredient data
- Never give a verdict without calling validate_brand first
- Only ask ONE clarifying question per turn — the most critical one
- Do not ask questions already answered in the user's profile
- If the product is clearly safe AND the user has no relevant risk factors, say so confidently
- format_response is the ONLY way to return a reply — always call it last`;

function buildSystemWithProfile(profile: Record<string, unknown>): string {
  const parts: string[] = [];

  // Personal info (spread from Supabase profile column into root by storage.ts)
  if (profile.name) parts.push(`Name: ${profile.name}`);
  if (profile.age) parts.push(`Age: ${profile.age}`);
  if (profile.height) parts.push(`Height: ${profile.height}`);
  if (profile.weight) parts.push(`Weight: ${profile.weight}`);
  if (profile.blood_type) parts.push(`Blood type: ${profile.blood_type}`);

  // Food safety (Supabase food_safety column)
  const fs = profile.food_safety as Record<string, unknown> | undefined;
  if (fs) {
    const allergies = fs.allergies as string[] | undefined;
    if (Array.isArray(allergies) && allergies.length > 0 && !allergies.includes("None"))
      parts.push(`Known allergies: ${allergies.join(", ")}`);

    const dietaryType = fs.dietary_type as string[] | string | undefined;
    const dtArr = Array.isArray(dietaryType) ? dietaryType : dietaryType ? [dietaryType] : [];
    if (dtArr.length > 0 && !dtArr.includes("None"))
      parts.push(`Dietary type: ${dtArr.join(", ")}`);

    const medConditions = fs.medical_conditions as string[] | undefined;
    if (Array.isArray(medConditions) && medConditions.length > 0)
      parts.push(`Medical conditions: ${medConditions.join(", ")}`);

    const dietaryNeeds = fs.dietary_needs as string[] | undefined;
    if (Array.isArray(dietaryNeeds) && dietaryNeeds.length > 0)
      parts.push(`Dietary needs: ${dietaryNeeds.join(", ")}`);

    const sensitivities = fs.sensitivities as string[] | undefined;
    if (Array.isArray(sensitivities) && sensitivities.length > 0)
      parts.push(`Sensitivities: ${sensitivities.join(", ")}`);
  }

  // Preferences (Supabase preferences column)
  const prefs = profile.preferences as Record<string, unknown> | undefined;
  if (prefs) {
    const cuisines = prefs.favorite_cuisines as string[] | undefined;
    if (Array.isArray(cuisines) && cuisines.length > 0)
      parts.push(`Favorite cuisines: ${cuisines.join(", ")}`);
    if (prefs.spice_level) parts.push(`Spice preference: ${prefs.spice_level}`);
    if (prefs.price_range) parts.push(`Budget: ${prefs.price_range}`);
    if (prefs.eating_goal) parts.push(`Eating goal: ${prefs.eating_goal}`);
  }

  // Priority (Supabase priority column)
  const priority = profile.priority as Record<string, unknown> | undefined;
  if (priority) {
    const df = priority.decision_factor as string[] | string | undefined;
    const dfArr = Array.isArray(df) ? df : df ? [df] : [];
    if (dfArr.length > 0)
      parts.push(`Verification priorities: ${dfArr.join(", ")}`);
  }

  // Legacy / ad-hoc memory keys saved mid-conversation by save_user_memory
  const knownKeys = new Set([
    "id", "name", "age", "height", "weight", "bmi", "blood_type",
    "medical_history", "current_medications", "insurance", "onboarded",
    "profile", "food_safety", "preferences", "priority", "onboarding",
  ]);
  for (const [k, v] of Object.entries(profile)) {
    if (!knownKeys.has(k) && v !== undefined && v !== null && v !== "")
      parts.push(`${k}: ${Array.isArray(v) ? v.join(", ") : String(v)}`);
  }

  const base = loadBaseSystemPrompt();
  if (parts.length === 0) return base;
  return base + "\n\n--- User Health Profile ---\n" + parts.join("\n");
}

// ---------------------------------------------------------------------------
// Tool declarations (Google SDK format)
// ---------------------------------------------------------------------------

const tools: FunctionDeclaration[] = [
  {
    name: "get_user_profile",
    description:
      "Fetch the user's current stored health profile. Use mid-conversation to double-check allergies or conditions before making a safety recommendation.",
    parameters: { type: SchemaType.OBJECT, properties: {}, required: [] },
  },
  {
    name: "ask_user_question",
    description:
      "Ask the user ONE clarifying question before giving a verdict. Use when: (1) brand/restaurant name is unknown, (2) the product contains common allergens and no allergy info is on file, (3) the product is a supplement/medication and no medication list is on file, (4) the product is a cosmetic and no skin sensitivities are on file. Only ask the single most critical missing piece. After calling this tool, call save_user_memory to note what was asked, then call format_response with the question as the reply and NO validation_result.",
    parameters: {
      type: SchemaType.OBJECT,
      properties: {
        question: { type: SchemaType.STRING, description: "The exact question to show the user" },
        reason: { type: SchemaType.STRING, description: "Why this info is needed for safety analysis" },
        memory_key: { type: SchemaType.STRING, description: "The save_user_memory key this answer will be stored under" },
      },
      required: ["question", "reason", "memory_key"],
    },
  },
  {
    name: "save_user_memory",
    description:
      "Persist a fact learned about the user mid-conversation (e.g. a new allergy, condition, or preference). Saved for future sessions.",
    parameters: {
      type: SchemaType.OBJECT,
      properties: {
        key: { type: SchemaType.STRING, description: "snake_case key e.g. 'latex_allergy'" },
        value: { type: SchemaType.STRING, description: "The value to store" },
      },
      required: ["key", "value"],
    },
  },
  {
    name: "validate_brand",
    description:
      "Look up a brand or restaurant chain's legitimacy, certifications, health violations, FDA/USDA warnings, recalls, and safety record. Works for packaged food, beverages, supplements, medications, cosmetics, personal care, baby products, cleaning products, and restaurant chains. Always call this before giving a final verdict.",
    parameters: {
      type: SchemaType.OBJECT,
      properties: {
        brand_name: { type: SchemaType.STRING, description: "The brand name or restaurant chain to look up (required)" },
        product_name: { type: SchemaType.STRING, description: "Specific product or dish name" },
        product_category: {
          type: SchemaType.STRING,
          description: "Category: food, beverage, supplement, medication, cosmetic, personal_care, baby_product, cleaning_product, restaurant, medical_device",
        },
        ingredients_of_concern: {
          type: SchemaType.ARRAY,
          items: { type: SchemaType.STRING },
          description: "Specific ingredients to flag against the user's profile (allergens, drug interactions, etc.)",
        },
      },
      required: ["brand_name", "product_category"],
    },
  },
  {
    name: "format_response",
    description:
      "ALWAYS call this as your final step to return your reply. If you asked a clarifying question, omit validation_result. If you gave a verdict, include the validation result from validate_brand.",
    parameters: {
      type: SchemaType.OBJECT,
      properties: {
        reply: { type: SchemaType.STRING, description: "Your message to the user" },
        validation_result: {
          type: SchemaType.OBJECT,
          description: "Result from validate_brand, or omit if asking a question",
          properties: {
            brand_name: { type: SchemaType.STRING },
            product_name: { type: SchemaType.STRING },
            trust_score: { type: SchemaType.NUMBER },
            verdict: { type: SchemaType.STRING },
            certifications: { type: SchemaType.ARRAY, items: { type: SchemaType.STRING } },
            red_flags: { type: SchemaType.ARRAY, items: { type: SchemaType.STRING } },
            reviews_summary: { type: SchemaType.STRING },
          },
        },
      },
      required: ["reply"],
    },
  },
];


// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ValidationFactor {
  category: string;           // e.g. "Food & Product Safety"
  status: "pass" | "warn" | "fail";
  findings: string[];         // bullet points shown to user
  summary: string;            // 1-2 sentence context
  sources: { title: string; url: string }[];
}

export interface ValidationResult {
  brand_name: string;
  product_name: string;
  trust_score: number | null;
  verdict: string;
  certifications: string[];
  red_flags: string[];
  reviews_summary: string;
  factors: ValidationFactor[];
  sources: unknown[];
  stub?: boolean;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResult {
  reply: string;
  validation: ValidationResult | null;
  askedQuestion: boolean;
  logId?: string;
}

const MODEL_NAME = "gemini-2.5-flash";

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

export async function runChat(
  message: string,
  conversationHistory: ChatMessage[],
  userProfile: Record<string, unknown>,
  imageBase64?: string,
  imageMediaType?: string
): Promise<ChatResult> {
  const systemInstruction = buildSystemWithProfile(userProfile);
  const callId = newCallId();
  const startedAt = Date.now();

  const model = genAI.getGenerativeModel({
    model: MODEL_NAME,
    systemInstruction,
    tools: [{ functionDeclarations: tools }],
    // Force it to use tools (it must call format_response to finish)
    toolConfig: { functionCallingConfig: { mode: "AUTO" as any } },
  });

  // Build history in Gemini Content format (cap at 10 turns)
  const trimmedHistory = conversationHistory.slice(-10);
  const history: Content[] = trimmedHistory.map((m) => ({
    role: m.role === "assistant" ? "model" : "user",
    parts: [{ text: m.content }],
  }));

  const chat = model.startChat({ history });

  // Build the first user message parts
  const userParts: Part[] = [];
  if (imageBase64 && imageMediaType) {
    userParts.push({ inlineData: { data: imageBase64, mimeType: imageMediaType } });
  }
  userParts.push({ text: message || "Please analyze this product." });

  let finalReply = "";
  let finalValidation: ValidationResult | null = null;

  // ---------------------------------------------------------------------------
  // Agentic loop
  // ---------------------------------------------------------------------------
  let currentParts: Part[] = userParts;

  while (true) {
    const result = await chat.sendMessage(currentParts);
    const response = result.response;
    const candidate = response.candidates?.[0];

    if (!candidate) break;

    const responseParts = candidate.content.parts;
    const functionCalls = responseParts.filter((p) => p.functionCall);

    // No tool calls — model returned text directly (fallback)
    if (functionCalls.length === 0) {
      finalReply = response.text();
      break;
    }

    // Dispatch all tool calls, collect results
    const functionResponses: FunctionResponsePart[] = [];
    let shouldBreak = false;

    for (const part of responseParts) {
      if (!part.functionCall) continue;

      const { name, args } = part.functionCall;
      let responseData: unknown;

      switch (name) {
        case "get_user_profile": {
          responseData = await getUser();
          break;
        }
        case "ask_user_question": {
          const a = args as { question: string; reason?: string };
          responseData = { noted: true, question: a.question };
          break;
        }
        case "save_user_memory": {
          const a = args as { key: string; value: string };
          await saveUserMemory(a.key, a.value);
          responseData = { saved: true, key: a.key };
          break;
        }
        case "validate_brand": {
          const a = args as { brand_name: string; product_name?: string; product_category?: string; ingredients_of_concern?: string[] };
          const result = await validateBrand(a.brand_name, a.product_name, a.product_category, a.ingredients_of_concern);
          finalValidation = result;
          responseData = result;
          break;
        }
        case "format_response": {
          const a = args as { reply: string; validation_result?: ValidationResult | null };
          finalReply = a.reply;
          // Don't overwrite finalValidation — it was already set by validate_brand
          // with full Tavily data (factors, sources, etc.) that Gemini can't reconstruct
          if (finalValidation === null && a.validation_result !== undefined) {
            finalValidation = a.validation_result ?? null;
          }
          responseData = { ok: true };
          shouldBreak = true;
          break;
        }
        default:
          responseData = { error: `Unknown tool: ${name}` };
      }

      functionResponses.push({
        functionResponse: { name, response: { result: responseData } },
      });
    }

    if (shouldBreak) break;

    // Feed tool results back for next loop iteration
    currentParts = functionResponses;
  }

  // Fire-and-forget log to ClickHouse so Autoval can inspect this call later.
  // No await — we don't want logging to slow the user-facing response.
  const latencyMs = Date.now() - startedAt;
  void logLlmCall({
    id: callId,
    input: message || "(image)",
    output: finalReply,
    model: MODEL_NAME,
    latency_ms: latencyMs,
  });

  return {
    reply: finalReply,
    validation: finalValidation,
    askedQuestion: finalValidation === null,
    logId: callId,
  };
}
