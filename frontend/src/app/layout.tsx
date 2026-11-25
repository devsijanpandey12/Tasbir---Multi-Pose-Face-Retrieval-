import type { Metadata } from "next";
import { Plus_Jakarta_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const sans = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Tasbir — Multi-Angle Event & Group Photo Finder",
  description:
    "Upload event or group photos once, then find every picture of yourself using 3 reference angles or a short head-turn video.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} dark`}>
      <body className="min-h-screen bg-[#0a0c10] text-zinc-100 antialiased selection:bg-amber-400/30 selection:text-amber-200">
        {children}
      </body>
    </html>
  );
}
