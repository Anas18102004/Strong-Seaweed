export type KappaphycusPrediction = {
  input: { lat: number; lon: number };
  nearest_grid: { lat: number; lon: number; distance_km: number };
  kappaphycus: {
    probability: number;
    probability_percent: number;
    raw_probability: number;
    pred_label: number;
    priority: "high" | "medium" | "low";
  };
  model: { release_tag: string; threshold: number };
  note: string;
};

const BASE_URL = import.meta.env.VITE_MODEL_API_URL || "http://127.0.0.1:8000";

export async function predictKappaphycus(lat: number, lon: number): Promise<KappaphycusPrediction> {
  const response = await fetch(`${BASE_URL}/predict/kappaphycus`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lat, lon }),
  });
  if (!response.ok) {
    const msg = await response.text();
    throw new Error(`Prediction API failed (${response.status}): ${msg}`);
  }
  return response.json();
}
