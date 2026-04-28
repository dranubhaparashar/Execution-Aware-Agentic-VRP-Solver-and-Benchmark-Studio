# VRP User Story — 4 Deliverables

A real day at a Dallas-area dispatch center. 20 technicians out of Depot Richardson handling 150 orders.
18 things go sideways across the day. 8 algorithms try to handle them.
Every scenario gets: setup narrative · winner's route map · 8-backend data table · 8-backend map gallery · why-we-test / what-happened / how-they-won.

## Files

| File | Format | Use it for |
|---|---|---|
| **`VRP_User_Story.html`** | Single 2.8MB HTML page (long-scroll) | Reading the story end-to-end. All 18 scenarios + 144 maps inline. No internet needed. |
| **`VRP_User_Story.pdf`** | 37-page PDF (888KB) | Print, share, email, attach. Every scenario fits cleanly on its own pages. |
| **`VRP_User_Story_Deck.html`** | Slide deck (351KB) | Live presentation. 1 slide per scenario, arrow-key nav, footer with progress bar. |
| **`VRP_Hub/`** folder | Notion-style multi-page hub | Browsing one scenario at a time. Index page links to 18 sub-pages. Each sub-page has prev/next pagination. |

## Notion-style hub

Open `VRP_Hub/index.html` to start. From there click any scenario card to navigate.
Each scenario page has:
- Breadcrumb (Hub / Scenario)
- Hero with emoji + title + key
- 🎬 Setup narrative (real-world context)
- 🗺️ Winner's route map (highlighted)
- 📊 8-backend benchmark data table
- 🖼️ All 8 algorithm maps (mini-gallery)
- 🤔 Why we test this · ⚡ What happened · 🎯 How [winner] won
- Pagination (prev / next scenario)

## Slide deck navigation

Open `VRP_User_Story_Deck.html` and use:
- `→` / `Space` / Next button — next slide
- `←` / Previous button — previous slide
- URL hash (`#0`, `#1`...) — bookmark a specific slide

20 slides total: 1 title + 18 scenarios + 1 outro.

## Key insights surfaced in the story

- **OR-Tools EA wins 12 of 18 scenarios** by lowest distance among zero-violation backends. OR-Tools (plain) wins 5. Esri wins 1.
- **PyVRP** consistently produces the lowest distance (~110-118 km) but routinely violates time windows (7K-13K late minutes). Distance-blind.
- **Greedy + 2-opt** runs fastest (~1.4s) but produces 12K+ late minutes on every scenario. Time-blind.
- **OSRM** uses real road distances but loses to time-blind heuristics on overall quality. ~13K late minutes.
- **Adaptive and Hybrid** produce zero violations but at the cost of 1.7-2× the distance of OR-Tools (~424 km vs 232 km).

## Editing

If you want to tweak content:
- **HTML user story**: edit `VRP_User_Story.html` directly — just text/HTML in dark-themed CSS
- **Hub pages**: edit individual `VRP_Hub/<scenario>.html` files
- **Slide deck**: edit `VRP_User_Story_Deck.html`

The dark midnight + mint palette is consistent across all 4 outputs.
