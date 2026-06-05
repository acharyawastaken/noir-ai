const express = require("express");
const { createProxyMiddleware } = require("http-proxy-middleware");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;
const API_URL = process.env.API_URL || "http://localhost:8000";

// Proxy API routes to FastAPI
// Using app.use() at root with pathFilter so Express doesn't strip the prefix
const apiProxy = createProxyMiddleware({
  target: API_URL,
  changeOrigin: true,
  pathFilter: ["/upload", "/query", "/reset", "/login"],
  logger: console,
});

app.use(apiProxy);

// Serve static frontend files
app.use(express.static(path.join(__dirname, "public")));

// SPA fallback
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.listen(PORT, () => {
  console.log(`\n  ✦  Frontend server running at http://localhost:${PORT}`);
  console.log(`  ↳  Proxying API requests to ${API_URL}\n`);
});
