import mongoose from "mongoose";

const chatSessionSchema = new mongoose.Schema(
  {
    userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true, index: true },
    title: { type: String, default: "New chat" },
    lastMessageAt: { type: Date, default: Date.now, index: true },
  },
  { timestamps: true },
);

export const ChatSession = mongoose.model("ChatSession", chatSessionSchema);
