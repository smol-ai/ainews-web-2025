import type { Metadata, Site, Socials } from "@types";

export const SITE: Site = {
  TITLE: "AINews with Smol.ai",
  DESCRIPTION: "Weekday recaps of top News for AI Engineers",
  EMAIL: "swyx@smol.ai",
  NUM_POSTS_ON_HOMEPAGE: 5,
  NUM_PROJECTS_ON_HOMEPAGE: 3,
};

export const HOME: Metadata = {
  TITLE: "Home",
  DESCRIPTION: "Weekday recaps of top News for AI Engineers",
};

export const ISSUES: Metadata = {
  TITLE: "Issues",
  DESCRIPTION: "A collection of articles on topics I am passionate about.",
};

export const PROJECTS: Metadata = {
  TITLE: "Projects",
  DESCRIPTION:
    "A collection of my projects with links to repositories and live demos.",
};

export const SOCIALS: Socials = [
  {
    NAME: "GitHub",
    HREF: "https://github.com/smol-ai",
  },
  {
    NAME: "X (@smol_ai)",
    HREF: "https://x.com/smol_ai",
  },
];
