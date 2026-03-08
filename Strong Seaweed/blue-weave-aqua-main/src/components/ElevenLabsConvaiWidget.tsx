import { createElement, useEffect } from "react";

const SCRIPT_SRC = "https://unpkg.com/@elevenlabs/convai-widget-embed";
const AGENT_ID = "agent_6901kk6n1rbxfcas4vg04hrz4s1z";

function ensureWidgetScript() {
  const found = document.querySelector(`script[src="${SCRIPT_SRC}"]`);
  if (found) return;
  const script = document.createElement("script");
  script.src = SCRIPT_SRC;
  script.async = true;
  script.type = "text/javascript";
  document.body.appendChild(script);
}

export default function ElevenLabsConvaiWidget() {
  useEffect(() => {
    ensureWidgetScript();
  }, []);

  return (
    <div className="fixed right-4 bottom-20 z-[70] md:bottom-6">
      {createElement("elevenlabs-convai", { "agent-id": AGENT_ID })}
    </div>
  );
}
