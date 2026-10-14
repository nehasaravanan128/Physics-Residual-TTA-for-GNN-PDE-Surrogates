"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, Activity, Cpu, BarChart2, Database, Waves } from "lucide-react";

const navItems = [
  { href: "/", label: "Dashboard", icon: Home },
  { label: "SURROGATE MODEL", section: true },
  { href: "/inference", label: "Flow Inference", icon: Activity },
  { href: "/train", label: "Model Training", icon: Cpu },
  { href: "/evaluate", label: "Verification & Eval", icon: BarChart2 },
  { label: "CONTINUUM DATA", section: true },
  { href: "/explore", label: "Dataset Explorer", icon: Database },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <nav className="sidebar">
      <div className="sidebar-logo">
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
          <Waves size={18} color="var(--accent-terracotta)" />
          <h1>MeshGraphNet</h1>
        </div>
        <p>Continuum Surrogate</p>
      </div>

      <div className="sidebar-nav">
        {navItems.map((item, i) => {
          if ("section" in item) {
            return (
              <span key={i} className="nav-section-label">
                {item.label}
              </span>
            );
          }
          const isActive = pathname === item.href;
          const Icon = item.icon!;
          return (
            <Link
              key={item.href}
              href={item.href!}
              className={`nav-item ${isActive ? "active" : ""}`}
            >
              <Icon size={15} style={{ opacity: isActive ? 1 : 0.7, color: isActive ? "var(--accent-terracotta)" : "inherit" }} />
              <span>{item.label}</span>
            </Link>
          );
        })}

        <div style={{ flex: 1 }} />
        <div
          style={{
            padding: "16px 22px",
            borderTop: "1px solid var(--border)",
            fontSize: 11,
            color: "var(--text-muted)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <span>Port 3001</span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 5, color: "var(--accent-sage)", fontWeight: 600 }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--accent-sage)" }} />
            Connected
          </span>
        </div>
      </div>
    </nav>
  );
}
