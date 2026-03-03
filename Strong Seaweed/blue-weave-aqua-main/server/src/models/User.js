import mongoose from "mongoose";

const userSchema = new mongoose.Schema(
  {
    name: { type: String, required: true, trim: true },
    email: { type: String, required: true, unique: true, lowercase: true, trim: true },
    passwordHash: { type: String, required: true },
    phone: { type: String, default: "" },
    state: { type: String, default: "" },
    role: { type: String, default: "" },
    preferences: {
      notifications: {
        predictionCompleted: { type: Boolean, default: true },
        riskAlerts: { type: Boolean, default: true },
        seasonalAdvisories: { type: Boolean, default: true },
        reportGenerated: { type: Boolean, default: true },
        newModelVersion: { type: Boolean, default: true },
      },
      dataModels: {
        proMode: { type: Boolean, default: false },
        aiExplanation: { type: Boolean, default: true },
      },
      appearance: {
        theme: { type: String, enum: ["light", "dark", "system"], default: "light" },
        confidenceBadge: { type: Boolean, default: true },
      },
    },
  },
  { timestamps: true },
);

export const User = mongoose.model("User", userSchema);
