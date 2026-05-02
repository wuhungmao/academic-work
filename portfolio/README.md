# Marvin Wu — Personal Portfolio

Live site: **https://wuhungmao.github.io/academic-work/**

Personal portfolio website for Marvin Wu (Wu Hung Mao), CS Specialist at the University of Toronto Mississauga and former Physical Layer (vDU) Developer Intern at Ericsson.

## Pages

| Route | Description |
|---|---|
| `/` | Home — typewriter intro, bio, Ericsson work experience, skills, contact |
| `/timeline` | Visual timeline of education, internship, and key projects (2021 – 2026) |
| `/courses` | All 300/400 level CS courses with grades and difficulty ratings |
| `/courses/:id` | Individual course page with reflection |
| `/projects` | 18 projects across Industry, AI/ML, Systems, Web, and Course categories |
| `/publications` | 2 co-authored research papers with citations and PDF links |
| `/hobbies` | Hobbies with photo albums (Hiking, Gaming, Cooking, Working Out) |

## Tech Stack

- **React 18** + **Vite** — frontend framework and build tool
- **React Router v6** (HashRouter) — client-side routing, compatible with GitHub Pages
- **CSS Modules** — per-component stylesheets, no CSS framework
- **Fira Code** — monospace font from Google Fonts
- **GitHub Actions** — automated build and deploy to GitHub Pages on every push to `main`

## Design

Terminal / hacker aesthetic: dark background (`#0d1117`), white accent (`#f0f6fc`), monospace font throughout. Inspired by GitHub's dark theme.

## Project Structure

```
portfolio/
├── public/
│   └── resume.pdf          # Place your resume PDF here for the download button
├── src/
│   ├── data/               # All content as plain JS files (easy to edit)
│   │   ├── courses.js
│   │   ├── hobbies.js
│   │   ├── projects.js
│   │   ├── publications.js
│   │   └── timeline.js
│   ├── components/
│   │   ├── Navbar.jsx
│   │   ├── TerminalWindow.jsx
│   │   └── PhotoAlbum.jsx
│   └── pages/
│       ├── Home.jsx
│       ├── Timeline.jsx
│       ├── Courses.jsx / CoursePage.jsx
│       ├── Projects.jsx
│       ├── Publications.jsx
│       ├── Hobbies.jsx / HobbyPage.jsx
├── vite.config.js          # base: '/academic-work/'
└── package.json
```

## Local Development

```bash
cd portfolio
npm install
npm run dev
```

Open http://localhost:5173/academic-work/

## Deployment

Deployment is fully automated via GitHub Actions (`.github/workflows/deploy.yml`).

Every push to `main` triggers a build and deploys `portfolio/dist/` to GitHub Pages. Hobby photos from `Personal website/Hobbies pics/` are copied into the build at deploy time — no duplicate files in the repo.

To update content, edit the files in `src/data/` and push to `main`.

## Adding Your Resume

Place your resume PDF at `portfolio/public/resume.pdf` and push to `main`. The "Download CV" button on the home page will then work automatically.
