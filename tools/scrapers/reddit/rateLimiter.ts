/**
 * Rate limiter with exponential backoff for API requests
 */
export class RateLimiter {
  private requestTimestamps: number[] = [];
  private maxRequestsPerMinute: number;
  private currentBackoffMs: number = 0;
  private maxBackoffMs: number;
  private failedAttempts: number = 0;
  private maxRetries: number = 10;

  constructor(maxRequestsPerMinute: number = 10, maxBackoffMinutes: number = 20) {
    this.maxRequestsPerMinute = maxRequestsPerMinute;
    this.maxBackoffMs = maxBackoffMinutes * 60 * 1000;
  }

  async throttle(): Promise<void> {
    // Clean up old timestamps (older than 1 minute)
    const now = Date.now();
    const oneMinuteAgo = now - 60 * 1000;
    this.requestTimestamps = this.requestTimestamps.filter(time => time > oneMinuteAgo);

    // If we're under the rate limit and no backoff is active
    if (this.requestTimestamps.length < this.maxRequestsPerMinute && this.currentBackoffMs === 0) {
      this.requestTimestamps.push(now);
      return;
    }

    // Calculate wait time based on rate limit
    let waitTime = 0;
    if (this.requestTimestamps.length >= this.maxRequestsPerMinute) {
      // Wait until the oldest request is more than a minute old
      waitTime = this.requestTimestamps[0] + 60 * 1000 - now;
    }

    // Apply exponential backoff if active
    waitTime = Math.max(waitTime, this.currentBackoffMs);

    if (waitTime > 0) {
      console.log(`Rate limit reached. Waiting ${Math.round(waitTime / 1000)} seconds before next request...`);
      await new Promise(resolve => setTimeout(resolve, waitTime));

      // After waiting, add the current time to timestamps
      this.requestTimestamps.push(Date.now());
    }
  }

  recordSuccess(): void {
    // Reset backoff on successful request
    if (this.failedAttempts > 0) {
      this.failedAttempts = 0;
      this.currentBackoffMs = 0;
    }
  }

  recordFailure(): boolean {
    this.failedAttempts++;

    // Exponential backoff calculation: 2^n seconds (converted to ms)
    // e.g., 2, 4, 8, 16, 32... seconds
    this.currentBackoffMs = Math.min(
      Math.pow(2, this.failedAttempts) * 1000,
      this.maxBackoffMs
    );

    console.log(`Request failed. Attempt ${this.failedAttempts}. Next retry in ${this.currentBackoffMs / 1000} seconds.`);

    // Return false if we've exceeded max retries
    return this.failedAttempts <= this.maxRetries;
  }

  // Getter for currentBackoffMs
  get backoffTime(): number {
    return this.currentBackoffMs;
  }

  // Getter for maxRequestsPerMinute
  get requestLimit(): number {
    return this.maxRequestsPerMinute;
  }

  // Getter for maxBackoffMs
  get maxBackoffTime(): number {
    return this.maxBackoffMs;
  }
}
