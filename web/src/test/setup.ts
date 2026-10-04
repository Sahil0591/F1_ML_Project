import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";
import { clearCache } from "../data/api";

afterEach(() => {
  cleanup();
  clearCache();
  vi.unstubAllGlobals();
});
