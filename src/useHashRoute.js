import { useEffect, useState } from "react";

// The page route lives in the URL hash as "#/<route>", so it works on GitHub Pages without server rewrites.
// Other hashes (e.g. "#info:..." note links, which never change the URL) are ignored.
function readRoute() {
  const m = window.location.hash.match(/^#\/([\w-]+)/);
  return m ? m[1] : null;
}

export function useHashRoute() {
  const [route, setRoute] = useState(readRoute);

  useEffect(() => {
    const onChange = () => setRoute(readRoute());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  return route;
}
