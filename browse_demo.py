"""Demo Playwright browsing session - searches Google, clicks links, scrolls."""
from playwright.sync_api import sync_playwright

def browse_session():
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    context = browser.new_context(viewport={'width': 1024, 'height': 1448})
    page = context.new_page()
    
    print("Navigating to Google...")
    page.goto('https://www.google.com')
    page.wait_for_load_state('networkidle')
    
    print("Searching for 'artificial intelligence'...")
    page.fill('textarea[name="q"]', 'artificial intelligence')
    page.press('textarea[name="q"]', 'Enter')
    page.wait_for_load_state('networkidle')
    
    print("Waiting for results...")
    page.wait_for_selector('div.g', timeout=5000)
    
    print("Clicking on second result...")
    results = page.query_selector_all('div.g')
    if len(results) >= 2:
        second_result = results[1]
        link = second_result.query_selector('a')
        if link:
            link.click()
            page.wait_for_load_state('networkidle')
    
    print("Scrolling down...")
    for i in range(3):
        page.mouse.wheel(0, 500)
        page.wait_for_timeout(1000)
    
    print("Scrolling back up...")
    page.evaluate('window.scrollTo(0, 0)')
    page.wait_for_timeout(1000)
    
    print("Taking screenshot...")
    page.screenshot(path='browsing_demo.png')
    
    print("Browsing session complete. Browser stays open for 60 seconds...")
    page.wait_for_timeout(60000)
    
    browser.close()
    pw.stop()

if __name__ == '__main__':
    browse_session()
