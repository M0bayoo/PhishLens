# PhishLens

A zero-friction Chrome extension that detects phishing pages in real time, entirely
on-device. Built as an MSc dissertation project.

PhishLens uses a two-phase pipeline: a lightweight structural classifier scores every
URL the moment a page loads, and a content-based check reads the live page to confirm
or override that score when brand identity is uncertain. No page content, URL, or
browsing history ever leaves the device.

**MSc Dissertation Project — Leeds Beckett University**
Author: John Oluwatobi Ogunbayo

## How it works

- **Phase 1 (structural):** A Random Forest trained on 34
  lexical and structural URL features, compiled directly to dependency-free
  JavaScript — no ONNX, no external ML runtime. Runs in under a millisecond.
- **Phase 2 (content):** For URLs Phase 1 can't confidently resolve, a content
  script reads the loaded page's computed style (not raw HTML) and checks it
  against a 2,013-entry brand dictionary for identity mismatches and
  credential-harvesting forms.
- **Verdict:** A single popup shows Green / Amber / Red with the reasoning behind
  it — no separate technical mode.

## Project structure

PhishLens-Extension/
├── manifest.json
├── background.js # gate() and fuse() decision logic
├── content.js # Phase 2 page-content check
├── popup.html / popup.js # verdict display
└── model/
├── forest_model_e.json
└── brands.json


## Dataset sources

| Source | Type | Role |
|---|---|---|
| PhiUSIIL | Phishing + Legitimate | Historical labelled base |
| PhishTank | Phishing only | Live, recently verified phishing URLs |
| Tranco Top 1M | Legitimate only | Diverse legitimate domains, weekly refresh |

## Setup

python3 -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
