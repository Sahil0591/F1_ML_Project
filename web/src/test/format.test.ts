import { describe, expect, it } from "vitest";
import { formatCoarse, formatProbability, formatSigned } from "../format";

describe("probability formatting", () => {
  it("never rounds a genuine small probability to 0.0%", () => {
    expect(formatProbability(0.0000457763671875, 65536)).toBe("0.0046%");
    expect(formatProbability(0.00099)).toBe("0.099%");
    expect(formatProbability(0.0009918212890625)).toBe("0.099%");
  });

  it("states zero draws explicitly", () => {
    expect(formatProbability(0, 65536)).toBe("0 of 65,536");
    expect(formatProbability(0)).toBe("0");
  });

  it("uses one decimal place above 0.1%", () => {
    expect(formatProbability(0.2347259521484375)).toBe("23.5%");
    expect(formatProbability(1)).toBe("100.0%");
  });

  it("keeps championship values to whole points without hiding tails", () => {
    expect(formatCoarse(0)).toBe("0 in sim.");
    expect(formatCoarse(0.004)).toBe("<1%");
    expect(formatCoarse(0.99407)).toBe(">99%");
    expect(formatCoarse(1)).toBe("100%");
    expect(formatCoarse(0.77123)).toBe("77%");
  });

  it("signs deltas with a true minus sign", () => {
    expect(formatSigned(3)).toBe("+3");
    expect(formatSigned(-2)).toBe("−2");
    expect(formatSigned(0)).toBe("0");
  });
});
