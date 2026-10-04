import { Navigate, Route, Routes } from "react-router-dom";
import { SiteHeader } from "./components/SiteHeader";
import { EmptyState, ErrorState, LoadingState } from "./components/States";
import { loadIndex } from "./data/api";
import { useResource } from "./data/useResource";
import { ChampionshipPage } from "./pages/ChampionshipPage";
import { PredictionsPage } from "./pages/PredictionsPage";
import { RacePage } from "./pages/RacePage";

export function App() {
  const index = useResource("index.json", loadIndex);
  const latestSeason = index.status === "ready" ? index.data.latest.season : null;
  return (
    <>
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <SiteHeader championshipSeason={latestSeason} />
      {index.status === "loading" ? (
        <main id="main" className="page">
          <LoadingState />
        </main>
      ) : null}
      {index.status === "error" ? (
        <main id="main" className="page">
          <ErrorState error={index.error} />
        </main>
      ) : null}
      {index.status === "ready" ? (
        <Routes>
          <Route
            path="/"
            element={
              <Navigate
                to={`/predictions/${index.data.latest.season}/${index.data.latest.round}`}
                replace
              />
            }
          />
          <Route path="/predictions" element={<PredictionsPage index={index.data} />} />
          <Route path="/predictions/:season/:round" element={<RacePage index={index.data} />} />
          <Route path="/championship/:season" element={<ChampionshipPage index={index.data} />} />
          <Route
            path="*"
            element={
              <main id="main" className="page">
                <EmptyState title="Page not found" />
              </main>
            }
          />
        </Routes>
      ) : null}
      <footer className="site-footer">
        <p>
          F1 ML Predictor is an independent research project. Development-only predictions, not
          validated forecasts. Not affiliated with Formula 1 or the FIA.
        </p>
      </footer>
    </>
  );
}
