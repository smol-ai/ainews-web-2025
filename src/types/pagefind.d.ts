interface Pagefind {
  filters(): Promise<Record<string, Record<string, number>>>;
}

declare global {
  interface Window {
    pagefind: Pagefind;
  }
}

export {}; 