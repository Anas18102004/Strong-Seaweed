import { useId } from "react";

type BrandLogoProps = {
  size?: "sm" | "md" | "lg";
  showWordmark?: boolean;
  className?: string;
  textClassName?: string;
};

const sizeMap = {
  sm: { box: "w-8 h-8", text: "text-base" },
  md: { box: "w-10 h-10", text: "text-lg" },
  lg: { box: "w-12 h-12", text: "text-xl" },
} as const;

export default function BrandLogo({
  size = "md",
  showWordmark = true,
  className = "",
  textClassName = "",
}: BrandLogoProps) {
  const gradA = useId().replace(/:/g, "");
  const gradB = useId().replace(/:/g, "");
  const icon = sizeMap[size];

  return (
    <div className={`flex items-center gap-2.5 ${className}`}>
      <div
        className={`${icon.box} rounded-xl overflow-hidden shadow-[0_10px_24px_-14px_rgba(2,132,199,0.95)]`}
        aria-label="BlueWeave Logo"
      >
        <svg viewBox="0 0 64 64" className="w-full h-full" role="img" aria-hidden="true">
          <defs>
            <linearGradient id={gradA} x1="6" y1="6" x2="58" y2="58" gradientUnits="userSpaceOnUse">
              <stop offset="0" stopColor="#14D8FF" />
              <stop offset="0.55" stopColor="#1D8BFF" />
              <stop offset="1" stopColor="#0F5BD8" />
            </linearGradient>
            <linearGradient id={gradB} x1="8" y1="12" x2="56" y2="44" gradientUnits="userSpaceOnUse">
              <stop offset="0" stopColor="#EBFCFF" />
              <stop offset="1" stopColor="#B8E8FF" />
            </linearGradient>
          </defs>
          <rect x="0" y="0" width="64" height="64" rx="14" fill={`url(#${gradA})`} />
          <path
            d="M8 36c4 0 6-1.6 8-3.2 2.2-1.8 4.4-3.6 9.2-3.6 4.9 0 7.1 1.8 9.4 3.6 2.1 1.6 4.2 3.2 8.6 3.2s6.4-1.6 8.5-3.2c1.2-.9 2.4-1.9 4.3-2.6v6.7c-1.4.6-2.4 1.4-3.5 2.2-2.2 1.7-4.7 3.7-9.3 3.7s-7.2-2-9.4-3.7c-2.1-1.7-4.1-3.1-8.6-3.1-4.4 0-6.4 1.4-8.6 3.1-2.2 1.7-4.7 3.7-9.4 3.7V36z"
            fill={`url(#${gradB})`}
            fillOpacity="0.95"
          />
          <path
            d="M8 25.5c4 0 6-1.6 8-3.2 2.2-1.8 4.4-3.6 9.2-3.6 4.9 0 7.1 1.8 9.4 3.6 2.1 1.6 4.2 3.2 8.6 3.2s6.4-1.6 8.5-3.2c1.1-.9 2.4-1.9 4.3-2.6v5.8c-1.5.6-2.5 1.4-3.6 2.2-2.2 1.7-4.6 3.7-9.2 3.7s-7.2-2-9.4-3.7c-2.2-1.7-4.2-3.1-8.7-3.1-4.4 0-6.4 1.4-8.6 3.1-2.2 1.7-4.7 3.7-9.4 3.7v-5.9z"
            fill="#FFFFFF"
            fillOpacity="0.75"
          />
          <circle cx="49.5" cy="17" r="4.2" fill="#F7FDFF" fillOpacity="0.95" />
        </svg>
      </div>
      {showWordmark && (
        <span className={`${icon.text} font-bold tracking-tight text-foreground whitespace-nowrap ${textClassName}`}>
          BlueWeave<span className="gradient-text"> AI</span>
        </span>
      )}
    </div>
  );
}
