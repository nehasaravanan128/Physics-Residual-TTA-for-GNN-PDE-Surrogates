import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "./components/Sidebar";

export const metadata: Metadata = {
  title: "MeshGraphNet Dashboard — Physics Surrogate",
  description: "Interactive dashboard for the MeshGraphNet GNN surrogate model — run inference, train, evaluate, and explore Burgers' equation and CylinderFlow datasets.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="app-layout">
          <Sidebar />
          <main className="main-content">{children}</main>
        </div>
      </body>
    </html>
  );
}
