/**
 * A lightweight youtube embed. Still should feel the same to the user, just MUCH faster to initialize and paint.
 *
 * Thx to these as the inspiration
 *   https://storage.googleapis.com/amp-vs-non-amp/youtube-lazy.html
 *   https://autoplay-youtube-player.glitch.me/
 *
 * Once built it, I also found these:
 *   https://github.com/ampproject/amphtml/blob/master/extensions/amp-youtube (👍👍)
 *   https://github.com/Daugilas/lazyYT
 *   https://github.com/vb/lazyframe
 */
class LiteYTEmbed extends HTMLElement {
    constructor() {
        super();

        // Gotta encode the untrusted value
        // https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html#rule-2---attribute-escape-before-inserting-untrusted-data-into-html-common-attributes
        this.videoId = encodeURIComponent(this.getAttribute('videoid'));
        this.videoTitle = this.getAttribute('title') || 'YouTube Video';
        this.videoPlay = this.getAttribute('playlabel') || 'Play';
        this.videoParams = this.getAttribute('params') || '';

        // Use a specific poster image if provided, otherwise use the standard YouTube thumbnail
        this.posterUrl = this.getAttribute('poster') || `https://i.ytimg.com/vi/${this.videoId}/hqdefault.jpg`;

        // Set up play button and other necessary UI
        this.setupUI();

        // Setup listeners
        this.addEventListener('pointerover', this.warmConnections.bind(this), {once: true});
        this.addEventListener('click', this.addIframe.bind(this));
    }

    setupUI() {
        // If we've already set up the UI, don't do it again
        if (this.querySelector('.lty-playbtn')) return;

        // If there isn't an explicit style set, use the poster image as the background
        if (!this.style.backgroundImage) {
            this.style.backgroundImage = `url("${this.posterUrl}")`;
        }

        // Create the play button if it doesn't exist
        if (!this.querySelector('.lty-playbtn')) {
            const playBtn = document.createElement('button');
            playBtn.type = 'button';
            playBtn.classList.add('lty-playbtn');
            playBtn.title = this.videoPlay;
            this.append(playBtn);

            // Add a visually-hidden label for screen readers
            const screenReaderLabel = document.createElement('span');
            screenReaderLabel.classList.add('lyt-visually-hidden');
            screenReaderLabel.textContent = this.videoPlay;
            playBtn.appendChild(screenReaderLabel);
        }

        // Add a title if one is provided
        if (this.getAttribute('title')) {
            const titleEl = document.createElement('div');
            titleEl.classList.add('lyt-title');
            titleEl.textContent = this.getAttribute('title');
            this.appendChild(titleEl);
        }
    }

    /**
     * Begin preconnecting to warm up the iframe load
     * Since the embed's network requests load within its iframe,
     *   preload/prefetch won't work and we need to use preconnect instead
     */
    warmConnections() {
        if (LiteYTEmbed.preconnected) return;

        // The iframe document and most of its subresources come from youtube.com
        LiteYTEmbed.addPrefetch('preconnect', 'https://www.youtube-nocookie.com');
        // The botguard script is fetched off from google.com
        LiteYTEmbed.addPrefetch('preconnect', 'https://www.google.com');

        // Not certain if these ad related domains are needed
        LiteYTEmbed.addPrefetch('preconnect', 'https://googleads.g.doubleclick.net');
        LiteYTEmbed.addPrefetch('preconnect', 'https://static.doubleclick.net');

        LiteYTEmbed.preconnected = true;
    }

    static addPrefetch(kind, url, as) {
        const linkEl = document.createElement('link');
        linkEl.rel = kind;
        linkEl.href = url;
        if (as) {
            linkEl.as = as;
        }
        document.head.append(linkEl);
    }

    async getYTPlayerInstance() {
        if (this.playerPromise) return await this.playerPromise;

        this.playerPromise = new Promise((resolve) => {
            // If we have an iframe already, resolve with its contentWindow.YT player
            const iframe = this.querySelector('iframe');
            if (iframe) {
                if (iframe.contentWindow?.YT?.Player) {
                    resolve(iframe.contentWindow.YT.Player);
                    return;
                }
            }

            // Otherwise, wait for iframe to be added first
            this.addEventListener('ly-playerready', () => {
                const iframe = this.querySelector('iframe');
                if (iframe?.contentWindow?.YT?.Player) {
                    resolve(iframe.contentWindow.YT.Player);
                }
            }, { once: true });

            // If there's no iframe yet, add one
            if (!iframe) this.addIframe();
        });

        return await this.playerPromise;
    }

    /**
     * Get a reference to the YT player instance
     * To use the Player API: https://developers.google.com/youtube/iframe_api_reference
     */
    async getYTPlayer() {
        if (this.ytPlayer) return this.ytPlayer;

        // Need to use a different approach to access the player API for this iframe
        const iframe = this.querySelector('iframe');
        if (!iframe) {
            // Add iframe if it doesn't exist
            this.addIframe();
            // Wait for the iframe to be ready
            await new Promise(resolve => {
                this.addEventListener('ly-playerready', resolve, { once: true });
            });
        }

        // The player might not be ready immediately, wait for it
        for (let i = 0; i < 10; i++) {
            if (iframe.contentWindow?.YT?.Player) {
                const playerInstance = iframe.contentWindow.YT.Player;
                this.ytPlayer = new playerInstance(iframe);
                break;
            }
            await new Promise(r => setTimeout(r, 100));
        }

        return this.ytPlayer;
    }

    addIframe() {
        if (this.classList.contains('lyt-activated')) return;

        this.classList.add('lyt-activated');

        const params = new URLSearchParams(this.videoParams || '');
        params.append('autoplay', '1');
        params.append('playsinline', '1');

        // Create iframe
        const iframeEl = document.createElement('iframe');
        iframeEl.title = this.videoTitle;
        iframeEl.width = 560;
        iframeEl.height = 315;
        iframeEl.frameBorder = '0';
        iframeEl.setAttribute('allow', 'accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture');
        iframeEl.setAttribute('allowfullscreen', '1');
        iframeEl.src = `https://www.youtube-nocookie.com/embed/${this.videoId}?${params.toString()}`;

        this.append(iframeEl);

        // Set up event listener to detect when the player is ready
        window.addEventListener('message', (e) => {
            if (e.source === iframeEl.contentWindow && e.data && e.data.event === 'onReady') {
                this.dispatchEvent(new CustomEvent('ly-playerready'));
            }
        });
    }
}

// Register the custom element
customElements.define('lite-youtube', LiteYTEmbed); 