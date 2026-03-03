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
  const gradC = useId().replace(/:/g, "");
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
              <stop offset="0" stopColor="#1AE4FF" />
              <stop offset="0.55" stopColor="#1D7FFF" />
              <stop offset="1" stopColor="#2046C8" />
            </linearGradient>
            <linearGradient id={gradB} x1="8" y1="12" x2="56" y2="44" gradientUnits="userSpaceOnUse">
              <stop offset="0" stopColor="#EEFDFF" />
              <stop offset="1" stopColor="#B8F0FF" />
            </linearGradient>
            <linearGradient id={gradC} x1="14" y1="18" x2="50" y2="50" gradientUnits="userSpaceOnUse">
              <stop offset="0" stopColor="#90DBFF" />
              <stop offset="1" stopColor="#D7F8FF" />
            </linearGradient>
          </defs>
          <rect width="64" height="64" rx="14" fill={`url(#${gradA})`} />
          <path
            d="M20 14h9.5c7.2 0 11.5 3.4 11.5 9.2 0 3.6-1.8 6.3-4.7 7.8 3.9 1.4 6.5 4.6 6.5 9.6 0 6.2-4.8 10.4-12.6 10.4H20V14zm9.1 14.1c3.4 0 5.3-1.5 5.3-4.2 0-2.5-1.9-3.9-5.3-3.9h-2.8v8.1h2.8zm1.3 16.8c3.9 0 6-1.6 6-4.8 0-3-2.2-4.6-6-4.6h-4.1v9.4h4.1z"
            fill={`url(#${gradB})`}
          />
          <path
            d="M45.6 13.5c2.2 0 4 1.8 4 4s-1.8 4-4 4a4 4 0 1 1 0-8zM50.5 23.8l-6.4 8.3 5.2 5.3-2.2 2.3-5.4-5.5-5.9 7.7-2.5-1.9 6.1-7.9-4.8-4.9 2.2-2.3 5 5.1 6.1-7.9 2.6 1.8z"
            fill={`url(#${gradC})`}
            fillOpacity="0.98"
          />
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
