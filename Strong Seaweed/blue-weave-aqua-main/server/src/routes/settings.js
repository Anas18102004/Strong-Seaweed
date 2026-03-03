import express from "express";
import bcrypt from "bcryptjs";
import { authRequired } from "../middleware/auth.js";
import { User } from "../models/User.js";

const router = express.Router();

function normalizePreferences(pref = {}) {
  return {
    notifications: {
      predictionCompleted: pref?.notifications?.predictionCompleted ?? true,
      riskAlerts: pref?.notifications?.riskAlerts ?? true,
      seasonalAdvisories: pref?.notifications?.seasonalAdvisories ?? true,
      reportGenerated: pref?.notifications?.reportGenerated ?? true,
      newModelVersion: pref?.notifications?.newModelVersion ?? true,
    },
    dataModels: {
      proMode: pref?.dataModels?.proMode ?? false,
      aiExplanation: pref?.dataModels?.aiExplanation ?? true,
    },
    appearance: {
      theme: pref?.appearance?.theme || "light",
      confidenceBadge: pref?.appearance?.confidenceBadge ?? true,
    },
  };
}

router.get("/", authRequired, async (req, res) => {
  const user = await User.findById(req.user.id).lean();
  if (!user) return res.status(404).json({ error: "user not found" });

  return res.json({
    profile: {
      name: user.name || "",
      email: user.email || "",
      phone: user.phone || "",
      state: user.state || "",
      role: user.role || "",
    },
    preferences: normalizePreferences(user.preferences),
    metadata: {
      modelVersion: process.env.MODEL_VERSION_LABEL || "Marine Core v2",
      updatedAt: user.updatedAt,
    },
  });
});

router.put("/profile", authRequired, async (req, res) => {
  const payload = req.body || {};
  const updates = {
    name: String(payload.name || "").trim(),
    phone: String(payload.phone || "").trim(),
    state: String(payload.state || "").trim(),
    role: String(payload.role || "").trim(),
  };
  if (!updates.name) return res.status(400).json({ error: "name is required" });

  const user = await User.findByIdAndUpdate(req.user.id, updates, { new: true }).lean();
  if (!user) return res.status(404).json({ error: "user not found" });

  return res.json({
    profile: {
      name: user.name || "",
      email: user.email || "",
      phone: user.phone || "",
      state: user.state || "",
      role: user.role || "",
    },
    metadata: {
      updatedAt: user.updatedAt,
    },
  });
});

router.put("/security/password", authRequired, async (req, res) => {
  const currentPassword = String(req.body?.currentPassword || "");
  const newPassword = String(req.body?.newPassword || "");
  if (!currentPassword || !newPassword) {
    return res.status(400).json({ error: "currentPassword and newPassword are required" });
  }
  if (newPassword.length < 8) {
    return res.status(400).json({ error: "new password must be at least 8 characters" });
  }

  const user = await User.findById(req.user.id);
  if (!user) return res.status(404).json({ error: "user not found" });

  const ok = await bcrypt.compare(currentPassword, user.passwordHash);
  if (!ok) return res.status(401).json({ error: "current password is incorrect" });

  user.passwordHash = await bcrypt.hash(newPassword, 10);
  await user.save();
  return res.json({ success: true });
});

router.put("/preferences", authRequired, async (req, res) => {
  const raw = req.body?.preferences || {};
  const currentUser = await User.findById(req.user.id);
  if (!currentUser) return res.status(404).json({ error: "user not found" });

  const current = normalizePreferences(currentUser.preferences || {});
  const next = {
    notifications: {
      predictionCompleted:
        raw?.notifications?.predictionCompleted ?? current.notifications.predictionCompleted,
      riskAlerts: raw?.notifications?.riskAlerts ?? current.notifications.riskAlerts,
      seasonalAdvisories:
        raw?.notifications?.seasonalAdvisories ?? current.notifications.seasonalAdvisories,
      reportGenerated: raw?.notifications?.reportGenerated ?? current.notifications.reportGenerated,
      newModelVersion: raw?.notifications?.newModelVersion ?? current.notifications.newModelVersion,
    },
    dataModels: {
      proMode: raw?.dataModels?.proMode ?? current.dataModels.proMode,
      aiExplanation: raw?.dataModels?.aiExplanation ?? current.dataModels.aiExplanation,
    },
    appearance: {
      theme: ["light", "dark", "system"].includes(raw?.appearance?.theme)
        ? raw.appearance.theme
        : current.appearance.theme,
      confidenceBadge: raw?.appearance?.confidenceBadge ?? current.appearance.confidenceBadge,
    },
  };

  currentUser.preferences = next;
  await currentUser.save();

  return res.json({
    preferences: normalizePreferences(currentUser.preferences || {}),
    metadata: { updatedAt: currentUser.updatedAt },
  });
});

export default router;
