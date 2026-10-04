import { useEffect, useState } from "react";
import { LoadError } from "./api";

export type Resource<T> =
  | { status: "loading" }
  | { status: "error"; error: LoadError }
  | { status: "ready"; data: T };

/** Load one document; `key` changes restart the load and drop stale responses. */
export function useResource<T>(key: string | null, loader: () => Promise<T>): Resource<T> {
  const [state, setState] = useState<{ key: string | null; value: Resource<T> }>({
    key,
    value: { status: "loading" },
  });
  useEffect(() => {
    if (key === null) return;
    let active = true;
    loader().then(
      (data) => active && setState({ key, value: { status: "ready", data } }),
      (error: unknown) =>
        active &&
        setState({
          key,
          value: {
            status: "error",
            error:
              error instanceof LoadError
                ? error
                : new LoadError("malformed", key, error instanceof Error ? error.message : String(error)),
          },
        }),
    );
    return () => {
      active = false;
    };
    // The loader is keyed by `key`; callers pass a fresh closure every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  if (key === null || state.key !== key) return { status: "loading" };
  return state.value;
}
