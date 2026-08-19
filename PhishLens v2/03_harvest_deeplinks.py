# Harvests real legitimate deep-link URLs from live Tranco-ranked websites.
#
# The source datasets are biased: Tranco supplies bare homepages (0% have a
# path) while live PhishTank URLs are 67.8% deep links. A model trained on
# this alone learns "has a path -> phishing", a dataset-construction
# artefact that misclassifies ordinary legitimate deep links in deployment.
# This step supplies the missing class of examples.
#
# A desktop Chrome user-agent string is used because commercial bot
# protection on legitimate sites suppresses content served to
# self-identified automated clients, which would bias collection toward
# unprotected sites. One-hop passive fetch only; no link-following and no
# form submission.
#
# A high skip rate is expected. Tranco ranks domains by DNS traffic, so its
# upper ranks contain CDN, DNS and ad-infrastructure hosts that serve no
# browsable page in any browser. Around 600-800 harvested links is
# sufficient: the goal is that "has a path" ceases to be a perfect class
# separator, not parity with the phishing deep-link count.

import pandas as pd
import random
from pathlib import Path
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

HOW_MANY_SITES = 500
LINKS_PER_SITE = 4
TIMEOUT_MS = 20000
SAVE_EVERY = 25
random.seed(42)

REAL_BROWSER = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36")

OUTPUT = Path('data/processed/tranco_deeplinks.csv')
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


# Loads the domain, retrying with a www. prefix since Tranco lists bare
# domains and many hosts only resolve with it. True if a page loaded.
def load_page(page, domain):
    for address in (f"https://{domain}", f"https://www.{domain}"):
        try:
            page.goto(address, timeout=TIMEOUT_MS, wait_until='domcontentloaded')
            return True
        except Exception:
            continue
    return False


# Returns same-site links that point below the homepage
def collect_links(page):
    try:
        all_links = page.eval_on_selector_all(
            'a[href]', 'elements => elements.map(e => e.href)')
    except Exception:
        return []

    # The site actually landed on, which may differ after redirects
    this_site = urlparse(page.url).netloc.replace('www.', '')
    if not this_site:
        return []

    internal = []
    for link in set(all_links):
        if not link.startswith('http'):
            continue
        link_site = urlparse(link).netloc.replace('www.', '')
        if not link_site.endswith(this_site):
            continue
        if urlparse(link).path in ('', '/'):
            continue
        internal.append(link)

    return internal


def main():
    tranco = pd.read_csv('data/raw/tranco_top1m.csv', header=None,
                         names=['rank', 'domain'])
    sites = tranco['domain'].dropna().head(20000).sample(
        n=HOW_MANY_SITES, random_state=42).tolist()

    # Resume from a previous run if one was interrupted
    harvested = []
    already_done = set()
    if OUTPUT.exists():
        previous = pd.read_csv(OUTPUT)
        harvested = previous.to_dict('records')
        already_done = set(previous['site'].unique())
        print(f"Resuming: {len(harvested)} links from "
              f"{len(already_done)} sites already done\n")

    todo = [s for s in sites if s not in already_done]
    print(f"Visiting {len(todo)} sites\n")

    successes = 0
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=REAL_BROWSER,
            ignore_https_errors=True,
            viewport={'width': 1366, 'height': 768},
        )
        page = context.new_page()

        try:
            for position, site in enumerate(todo, 1):
                links = collect_links(page) if load_page(page, site) else []

                if not links:
                    print(f"[{position:3}/{len(todo)}] {site}: skipped")
                else:
                    chosen = random.sample(links, min(LINKS_PER_SITE, len(links)))
                    for link in chosen:
                        harvested.append({'URL': link, 'label': 1,
                                          'source': 'tranco_deeplink',
                                          'site': site})
                    successes += 1
                    print(f"[{position:3}/{len(todo)}] {site}: {len(chosen)} links "
                          f"(total {len(harvested)})")

                if position % SAVE_EVERY == 0:
                    pd.DataFrame(harvested).drop_duplicates('URL').to_csv(
                        OUTPUT, index=False)

        # Save whatever was collected before an interruption
        except KeyboardInterrupt:
            print("\nInterrupted. Saving progress...")

        browser.close()

    results = pd.DataFrame(harvested).drop_duplicates('URL')
    results.to_csv(OUTPUT, index=False)

    print(f"\n{'=' * 60}")
    print(f"Sites that worked : {successes}")
    print(f"Deep links saved  : {len(results)}")
    print(f"{'=' * 60}")

    if len(results) < 600:
        print("\nUnder 600 links. Re-run to resume and collect more.")
    else:
        print("\nNext: python 04_train_model.py")


if __name__ == '__main__':
    main()
