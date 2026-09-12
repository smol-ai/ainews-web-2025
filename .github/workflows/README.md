# Cloudflare deployment

The manually triggered workflow builds and deploys Worker `smol-news` in Cloudflare account
`2d017c943ff16e4c52783635ef05e535`. Configure the repository secret
`CLOUDFLARE_API_TOKEN` with permission to deploy that account's Workers.

Local deployment uses `pnpm run deploy`. Run `pnpm build` then `pnpm preview` to
exercise the output in workerd without watching the large archive.
Automatic publishing remains disabled until the token is configured. The Wrangler asset exclusions prevent server
code from being served as public files. Static issue pages, feeds and Pagefind
are assets; the OG endpoints run in the Worker with the existing React layouts.

The existing recent-content guard is retained. An intentional restoration of
an archived snapshot can use `BYPASS_RECENT_CONTENT_CHECK=true pnpm build`.
