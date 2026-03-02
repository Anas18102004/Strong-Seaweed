import mongoose from "mongoose";

const numberField = { type: Number, default: null };

const predictionSubmissionSchema = new mongoose.Schema(
  {
    userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true, index: true },
    lat: { type: Number, required: true },
    lon: { type: Number, required: true },
    locationName: { type: String, default: "" },
    season: { type: String, default: "" },
    depthM: numberField,
    overrides: {
      temperatureC: numberField,
      salinityPpt: numberField,
    },
    advanced: {
      ph: numberField,
      turbidityNtu: numberField,
      currentVelocityMs: numberField,
      waveHeightM: numberField,
      rainfallMm: numberField,
      tidalAmplitudeM: numberField,
    },
    prediction: { type: mongoose.Schema.Types.Mixed, required: true },
  },
  { timestamps: true },
);

predictionSubmissionSchema.index({ createdAt: -1 });
predictionSubmissionSchema.index({ userId: 1, createdAt: -1 });

export const PredictionSubmission = mongoose.model("PredictionSubmission", predictionSubmissionSchema);

