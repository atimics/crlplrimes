#!/usr/bin/env node

process.stdout.write(`${JSON.stringify({ metrics: { bridge_ready: 1 } })}\n`);
