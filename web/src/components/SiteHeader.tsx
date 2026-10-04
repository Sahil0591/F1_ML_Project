import { useState } from "react";
import { NavLink } from "react-router-dom";

interface SiteHeaderProps {
  championshipSeason: number | null;
}

export function SiteHeader({ championshipSeason }: SiteHeaderProps) {
  const [open, setOpen] = useState(false);
  const close = () => setOpen(false);
  return (
    <header className="site-header">
      <div className="site-header-inner">
        <NavLink to="/" className="brand" onClick={close}>
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-text">
            F1 ML <span>Predictor</span>
          </span>
        </NavLink>
        <button
          type="button"
          className="menu-toggle"
          aria-expanded={open}
          aria-controls="primary-nav"
          onClick={() => setOpen((value) => !value)}
        >
          <span className="visually-hidden">{open ? "Close menu" : "Open menu"}</span>
          <span className="menu-icon" aria-hidden="true" />
        </button>
        <nav id="primary-nav" aria-label="Primary" className={open ? "primary-nav open" : "primary-nav"}>
          <NavLink to="/predictions" onClick={close}>
            Predictions
          </NavLink>
          {championshipSeason !== null ? (
            <NavLink to={`/championship/${championshipSeason}`} onClick={close}>
              Championship
            </NavLink>
          ) : null}
        </nav>
      </div>
    </header>
  );
}
