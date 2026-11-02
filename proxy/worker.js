// Cloudflare Worker that relays table-hockey.eu API reads for the simulator website.
// Deploy it, then build the site with VITE_TH_PROXY="https://<your-worker>.workers.dev/?url=".
const ALLOWED = "https://table-hockey.eu/api/";

export default {
  async fetch(request) {
    const cors = { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "Accept" };
    if (request.method === "OPTIONS") return new Response(null, { headers: cors });
    const target = new URL(request.url).searchParams.get("url") ?? "";
    if (!target.startsWith(ALLOWED)) return new Response("Only table-hockey.eu API URLs are relayed", { status: 400, headers: cors });
    const r = await fetch(target, { headers: { Accept: "application/json" }, cf: { cacheTtl: 30 } });
    return new Response(r.body, { status: r.status, headers: { ...cors, "Content-Type": r.headers.get("Content-Type") ?? "application/json" } });
  },
};
