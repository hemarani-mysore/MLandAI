// @ts-check
const { defineConfig } = require("@playwright/test");

module.exports = defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  // execution_agent.py overrides this with `--reporter=json` (printed to stdout) when
  // it runs a spec programmatically; "list" is just a sane default for manual runs.
  reporter: "list",
  use: {
    headless: true,
    trace: "retain-on-failure",
  },
});
