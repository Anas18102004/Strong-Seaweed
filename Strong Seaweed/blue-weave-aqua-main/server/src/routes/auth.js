import express from "express";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import { User } from "../models/User.js";
import { config } from "../config.js";
import { authRequired } from "../middleware/auth.js";

const router = express.Router();

function createToken(user) {
  return jwt.sign({ id: user._id.toString(), email: user.email, name: user.name, role: user.role || "" }, config.jwtSecret, {
    expiresIn: "7d",
  });
}

router.post("/signup", async (req, res) => {
  try {
    const { name, email, password, phone = "", state = "", role = "" } = req.body || {};
    if (!name || !email || !password) return res.status(400).json({ error: "name, email and password are required" });
    if (password.length < 8) return res.status(400).json({ error: "password must be at least 8 characters" });

    const exists = await User.findOne({ email: String(email).toLowerCase().trim() });
    if (exists) return res.status(409).json({ error: "email already registered" });

    const passwordHash = await bcrypt.hash(password, 10);
    const user = await User.create({ name, email, passwordHash, phone, state, role });
    const token = createToken(user);

    res.status(201).json({
      token,
      user: { id: user._id.toString(), name: user.name, email: user.email, role: user.role || "" },
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "signup failed" });
  }
});

router.post("/signin", async (req, res) => {
  try {
    const { email, password } = req.body || {};
    if (!email || !password) return res.status(400).json({ error: "email and password are required" });

    const user = await User.findOne({ email: String(email).toLowerCase().trim() });
    if (!user) return res.status(401).json({ error: "invalid credentials" });

    const ok = await bcrypt.compare(password, user.passwordHash);
    if (!ok) return res.status(401).json({ error: "invalid credentials" });

    const token = createToken(user);
    res.json({
      token,
      user: { id: user._id.toString(), name: user.name, email: user.email, role: user.role || "" },
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "signin failed" });
  }
});

router.get("/me", authRequired, async (req, res) => {
  const user = await User.findById(req.user.id).lean();
  if (!user) return res.status(404).json({ error: "user not found" });

  res.json({ user: { id: user._id.toString(), name: user.name, email: user.email, role: user.role || "" } });
});

export default router;
