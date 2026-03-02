import express from "express";
import cors from "cors";
import mongoose from "mongoose";
import { config } from "./config.js";
import authRoutes from "./routes/auth.js";
import aiRoutes from "./routes/ai.js";
import predictRoutes from "./routes/predict.js";

const app = express();

function isDevLocalOrigin(origin) {
  return /^https?:\/\/(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})(:\d+)?$/i.test(origin);
}

function isIpv4Origin(origin) {
  return /^https?:\/\/(\d{1,3}\.){3}\d{1,3}(:\d+)?$/i.test(origin);
}

app.use(
  cors({
    origin(origin, callback) {
      if (!origin) return callback(null, true);
      if (config.corsOrigins.includes(origin)) return callback(null, true);
      if (isDevLocalOrigin(origin)) return callback(null, true);
      if (isIpv4Origin(origin)) return callback(null, true);
      return callback(new Error("CORS origin not allowed"));
    },
    credentials: false,
  }),
);
app.use(express.json({ limit: "1mb" }));

app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", service: "blueweave-backend" });
});

app.use("/api/auth", authRoutes);
app.use("/api/ai", aiRoutes);
app.use("/api/predict", predictRoutes);

async function start() {
  await mongoose.connect(config.mongoUri);
  app.listen(config.port, () => {
    console.log(`Backend running on http://127.0.0.1:${config.port}`);
  });
}

start().catch((err) => {
  console.error("Backend failed to start", err);
  process.exit(1);
});
