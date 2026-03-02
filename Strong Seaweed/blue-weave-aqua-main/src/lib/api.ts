const API_BASE_URL = import.meta.env.VITE_API_URL || "";

export type AuthUser = {
  id: string;
  name: string;
  email: string;
  role?: string;
};

export type AuthResponse = {
  token: string;
  user: AuthUser;
};

export type ChatResponse = {
  answer: string;
  model: string;
  sessionId?: string;
  status?: string;
  provider?: string;
  routedAgent?: string;
  latencyMs?: number;
  cached?: boolean;
};

export type AgentResponse = {
  agent: string;
  answer: string;
  stack: string[];
  provider?: string;
  status?: string;
  latencyMs?: number;
  cached?: boolean;
};

export type VoiceResponse = {
  answer: string;
  ttsText: string;
  audioBase64?: string;
  audioMime?: string;
  voiceProvider?: string;
  voiceProfile?: VoiceProfile;
  model: string;
  sessionId?: string;
  status?: string;
  provider?: string;
  routedAgent?: string;
  latencyMs?: number;
  cached?: boolean;
};

export type VoiceProfile = "female" | "male" | "default";

export type AiProviderStatus = {
  provider: string;
  configured: boolean;
  online: boolean;
  detail: string;
  voiceSupported?: boolean;
};

export type AiStatusResponse = {
  routerMode: string;
  providers: AiProviderStatus[];
};

export type SpeciesScore = {
  speciesId: string;
  displayName: string;
  ready: boolean;
  probabilityPercent: number | null;
  priority: string;
  reason: string;
};

export type SpeciesPredictionResponse = {
  input: { lat: number; lon: number };
  source: string;
  modelRelease: string;
  nearestGrid: { lat: number; lon: number; distance_km: number } | null;
  species: SpeciesScore[];
  bestSpecies: SpeciesScore | null;
  warnings: string[];
};

export type PredictionFormInput = {
  locationName?: string;
  season?: string;
  depthM?: number | null;
  overrides?: {
    temperatureC?: number | null;
    salinityPpt?: number | null;
  };
  advanced?: {
    ph?: number | null;
    turbidityNtu?: number | null;
    currentVelocityMs?: number | null;
    waveHeightM?: number | null;
    rainfallMm?: number | null;
    tidalAmplitudeM?: number | null;
  };
};

export type ChatSession = {
  id: string;
  title: string;
  lastMessageAt: string;
  createdAt: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
};

export type PredictionSubmissionItem = {
  id: string;
  lat: number;
  lon: number;
  locationName: string;
  season: string;
  createdAt: string;
  bestSpecies: SpeciesScore | null;
};

type ChatStreamEvent =
  | { type: "delta"; token?: string }
  | {
      type: "done";
      sessionId?: string;
      model?: string;
      provider?: string;
      status?: string;
    };

async function request<T>(path: string, options: RequestInit = {}, token?: string): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string> | undefined),
  };
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed (${res.status})`);
  }

  return res.json();
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),

  signIn: (email: string, password: string) =>
    request<AuthResponse>("/api/auth/signin", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  signUp: (payload: {
    name: string;
    email: string;
    password: string;
    phone?: string;
    state?: string;
    role?: string;
  }) =>
    request<AuthResponse>("/api/auth/signup", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  me: (token: string) => request<{ user: AuthUser }>("/api/auth/me", { method: "GET" }, token),

  chat: (question: string, token?: string, sessionId?: string) =>
    request<ChatResponse>(
      "/api/ai/chat",
      {
        method: "POST",
        body: JSON.stringify({ question, sessionId }),
      },
      token,
    ),

  chatStream: async (
    question: string,
    token?: string,
    sessionId?: string,
    onDelta?: (tokenChunk: string) => void,
  ): Promise<ChatResponse> => {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (token) headers.Authorization = `Bearer ${token}`;

    const res = await fetch(`${API_BASE_URL}/api/ai/chat/stream`, {
      method: "POST",
      headers,
      body: JSON.stringify({ question, sessionId }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `Request failed (${res.status})`);
    }

    if (!res.body) {
      throw new Error("Streaming response not available in this browser.");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let answer = "";
    let donePayload: Partial<ChatResponse> = {};

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const events = buffer.split("\n\n");
      buffer = events.pop() || "";

      for (const evt of events) {
        const line = evt
          .split("\n")
          .find((l) => l.trimStart().startsWith("data:"));
        if (!line) continue;
        const payloadText = line.replace(/^data:\s*/, "");
        if (!payloadText) continue;

        let payload: ChatStreamEvent;
        try {
          payload = JSON.parse(payloadText);
        } catch {
          continue;
        }

        if (payload.type === "delta") {
          const chunk = String(payload.token || "");
          answer += chunk;
          onDelta?.(chunk);
        } else if (payload.type === "done") {
          donePayload = {
            sessionId: payload.sessionId,
            model: payload.model || "stream",
            provider: payload.provider,
            status: payload.status,
          };
        }
      }
    }

    return {
      answer: answer.trim() || "No response received.",
      model: String(donePayload.model || "stream"),
      sessionId: donePayload.sessionId,
      provider: donePayload.provider,
      status: donePayload.status,
    };
  },

  voiceRespond: (
    question: string,
    token?: string,
    sessionId?: string,
    locale = "en-US",
    voiceProfile: VoiceProfile = "female",
  ) =>
    request<VoiceResponse>(
      "/api/ai/voice/respond",
      {
        method: "POST",
        body: JSON.stringify({ question, sessionId, locale, voiceProfile }),
      },
      token,
    ),

  getChatSessions: (token?: string) => request<{ sessions: ChatSession[] }>("/api/ai/chat/sessions", { method: "GET" }, token),

  getChatMessages: (sessionId: string, token?: string) =>
    request<{ session: ChatSession; messages: ChatMessage[] }>(`/api/ai/chat/sessions/${sessionId}/messages`, { method: "GET" }, token),

  deleteChatSession: (sessionId: string, token?: string) =>
    request<{ success: boolean }>(`/api/ai/chat/sessions/${sessionId}`, { method: "DELETE" }, token),

  askAgent: (agent: string, question: string, token?: string) =>
    request<AgentResponse>(
      "/api/ai/agent",
      {
        method: "POST",
        body: JSON.stringify({ agent, question }),
      },
      token,
    ),

  aiStatus: (token?: string) => request<AiStatusResponse>("/api/ai/status", { method: "GET" }, token),

  predictSpecies: (lat: number, lon: number, token?: string, formInput?: PredictionFormInput) =>
    request<SpeciesPredictionResponse>(
      "/api/predict/species",
      {
        method: "POST",
        body: JSON.stringify({ lat, lon, formInput }),
      },
      token,
    ),

  mySubmissions: (token?: string, limit = 20) =>
    request<{ total: number; submissions: PredictionSubmissionItem[] }>(`/api/predict/submissions/me?limit=${limit}`, { method: "GET" }, token),
};

export { API_BASE_URL };

