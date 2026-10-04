import { teamColour } from "../teams";

export interface ChartSeries {
  id: string;
  label: string;
  colourKey: string | null;
  values: (number | null)[];
}

interface LineChartProps {
  title: string;
  xLabels: string[];
  series: ChartSeries[];
}

const WIDTH = 640;
const HEIGHT = 240;
const PAD = { top: 16, right: 120, bottom: 32, left: 44 };

/** Probability evolution; it supplements the tables and never replaces them. */
export function LineChart({ title, xLabels, series }: LineChartProps) {
  const plotWidth = WIDTH - PAD.left - PAD.right;
  const plotHeight = HEIGHT - PAD.top - PAD.bottom;
  const max = Math.max(0.05, ...series.flatMap((item) => item.values.map((value) => value ?? 0)));
  const top = Math.min(1, Math.ceil(max * 10) / 10);
  const x = (index: number) =>
    PAD.left + (xLabels.length === 1 ? plotWidth / 2 : (index * plotWidth) / (xLabels.length - 1));
  const y = (value: number) => PAD.top + plotHeight - (value / top) * plotHeight;
  const ticks = [0, top / 2, top];
  const summary = series
    .map((item) => {
      const last = [...item.values].reverse().find((value) => value !== null);
      return `${item.label} ${last === undefined || last === null ? "n/a" : `${Math.round(last * 100)}%`}`;
    })
    .join(", ");
  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`${title}. Latest values: ${summary}.`}
        preserveAspectRatio="xMidYMid meet"
      >
        {ticks.map((tick) => (
          <g key={tick}>
            <line
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={y(tick)}
              y2={y(tick)}
              className="chart-grid"
            />
            <text
              x={PAD.left - 8}
              y={y(tick)}
              className="chart-tick"
              textAnchor="end"
              dominantBaseline="middle"
            >
              {Math.round(tick * 100)}%
            </text>
          </g>
        ))}
        {xLabels.map((label, index) => (
          <text
            key={`${label}-${index}`}
            x={x(index)}
            y={HEIGHT - 10}
            className="chart-tick"
            textAnchor="middle"
          >
            {label}
          </text>
        ))}
        {series.map((item, seriesIndex) => {
          const points = item.values
            .map((value, index) => (value === null ? null : `${x(index)},${y(value)}`))
            .filter((point): point is string => point !== null);
          const lastIndex = item.values.reduce<number>(
            (found, value, index) => (value === null ? found : index),
            -1,
          );
          const last = lastIndex >= 0 ? (item.values[lastIndex] ?? null) : null;
          const colour = teamColour(item.colourKey);
          return (
            <g key={item.id}>
              <polyline
                points={points.join(" ")}
                fill="none"
                stroke={colour}
                strokeWidth={2}
                strokeDasharray={seriesIndex % 2 === 1 ? "5 3" : undefined}
              />
              {item.values.map((value, index) =>
                value === null ? null : (
                  <circle key={index} cx={x(index)} cy={y(value)} r={3} fill={colour} />
                ),
              )}
              {last !== null ? (
                <text
                  x={x(lastIndex) + 8}
                  y={y(last)}
                  className="chart-label"
                  dominantBaseline="middle"
                >
                  {item.label}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
