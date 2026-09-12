import 'react';

declare module 'react' {
  interface HTMLAttributes<T> {
    /** Tailwind styles interpreted by Satori when rendering social images. */
    tw?: string;
  }
}
