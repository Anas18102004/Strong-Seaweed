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
        aria-label="Akuara Logo"
      >
        <svg viewBox="0 0 64 64" className="w-full h-full" role="img" aria-hidden="true">
          <defs>
            <linearGradient id={gradA} x1="6" y1="6" x2="58" y2="58" gradientUnits="userSpaceOnUse">
              <stop offset="0" stopColor="#1AE4FF" />
              <stop offset="0.55" stopColor="#1D7FFF" />
              <stop offset="1" stopColor="#2046C8" />
            </linearGradient>
            <linearGradient id={gradB} x1="14" y1="10" x2="50" y2="54" gradientUnits="userSpaceOnUse">
              <stop offset="0" stopColor="#EEFDFF" />
              <stop offset="1" stopColor="#B8F0FF" />
            </linearGradient>
            <linearGradient id={gradC} x1="35" y1="15" x2="56" y2="31" gradientUnits="userSpaceOnUse">
              <stop offset="0" stopColor="#9FF6FF" />
              <stop offset="1" stopColor="#35D7FF" />
            </linearGradient>
          </defs>
          <rect width="64" height="64" rx="14" fill={`url(#${gradA})`} />
          <path
            d="M31.9 12.8 14.6 51h7.1l4-9h12.6l4 9h7.1L31.9 12.8Zm-3.3 22.9 3.3-7.6 3.3 7.6h-6.6Z"
            fill={`url(#${gradB})`}
          />
          <path
            d="M46.7 15.3a4.7 4.7 0 1 0 0 9.4 4.7 4.7 0 0 0 0-9.4Zm0 1.9a2.8 2.8 0 1 1 0 5.6 2.8 2.8 0 0 1 0-5.6ZM42.4 29.1c4.9-1.3 9.8 1.1 12.5 5.7l-1.8 1c-2.2-3.8-6.2-5.7-10.2-4.7l-.5-2Z"
            fill={`url(#${gradC})`}
            fillOpacity="0.98"
          />
        </svg>
      </div>
      {showWordmark && (
        <span className={`${icon.text} font-bold tracking-tight text-foreground whitespace-nowrap ${textClassName}`}>
          Akuara<span className="gradient-text"> AI</span>
        </span>
      )}
    </div>
  );
}
