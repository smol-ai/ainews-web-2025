export function parseTimeInput(input: string | undefined, base?: Date): Date | undefined {
  if (!input) return undefined;
  const s = String(input).trim();
  if (!s) return undefined;

  // First, try native Date parsing
  const native = new Date(s);
  if (!isNaN(native.getTime())) return native;

  // Relative shorthand: 3d, 1w, 12h, 90m, 45s; also long forms like days, hours, minutes, seconds
  const re = /^(-?\d+)\s*(w|wk|wks|week|weeks|d|day|days|h|hr|hrs|hour|hours|min|mins|m|minute|minutes|s|sec|secs|second|seconds)$/i;
  const m = s.toLowerCase().match(re);
  if (!m) return undefined;

  const n = parseInt(m[1], 10);
  const unit = m[2];
  let msPerUnit = 0;
  if (/(^w$|^wk$|^wks$|^week$|^weeks$)/.test(unit)) msPerUnit = 7 * 24 * 60 * 60 * 1000;
  else if (/(^d$|^day$|^days$)/.test(unit)) msPerUnit = 24 * 60 * 60 * 1000;
  else if (/(^h$|^hr$|^hrs$|^hour$|^hours$)/.test(unit)) msPerUnit = 60 * 60 * 1000;
  else if (/(^min$|^mins$|^m$|^minute$|^minutes$)/.test(unit)) msPerUnit = 60 * 1000;
  else if (/(^s$|^sec$|^secs$|^second$|^seconds$)/.test(unit)) msPerUnit = 1000;
  else return undefined;

  const anchor = base || new Date();
  // By default, positive N means "N units before the anchor"
  const delta = n * msPerUnit;
  return new Date(anchor.getTime() - delta);
}

export function resolveWindow(startInput: string | undefined, endInput: string | undefined): { start: Date; end: Date } {
  const now = new Date();
  const endParsed = parseTimeInput(endInput, now);
  if (endInput !== undefined && (!endParsed || isNaN(endParsed.getTime()))) throw new Error('Invalid end time; use an ISO date or a duration such as 3d.');
  const end = endParsed || now;
  const startParsed = parseTimeInput(startInput, end);
  if (startInput !== undefined && (!startParsed || isNaN(startParsed.getTime()))) throw new Error('Invalid start time; use an ISO date or a duration such as 3d.');
  const start = startParsed || new Date(end.getTime() - 24 * 60 * 60 * 1000);
  return { start, end };
}
