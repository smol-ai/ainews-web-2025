import fs from 'node:fs';
import path from 'node:path';
import { randomUUID, createHash } from 'node:crypto';

export function readJson<T>(file: string): T | undefined {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')) as T; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined; throw error; }
}

export function writeJson(file: string, value: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.${randomUUID()}.tmp`;
  const fd = fs.openSync(temporary, 'wx', 0o600);
  try { fs.writeFileSync(fd, JSON.stringify(value, null, 2)); fs.fsyncSync(fd); }
  finally { fs.closeSync(fd); }
  fs.renameSync(temporary, file);
}

/** Persist only successful operations. Failed calls remain retryable. */
export async function checkpoint<T>(file: string, operation: () => Promise<T>): Promise<T> {
  const saved = readJson<{ value: T }>(file);
  if (saved) return saved.value;
  const value = await operation();
  writeJson(file, { value });
  return value;
}

export function fingerprint(value: unknown): string {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

export function acquireLock(directory: string): () => void {
  const file = path.join(directory, 'run.lock');
  fs.mkdirSync(directory, { recursive: true });
  try { fs.writeFileSync(file, String(process.pid), { flag: 'wx', mode: 0o600 }); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
    const pid = Number(fs.readFileSync(file, 'utf8'));
    if (!Number.isSafeInteger(pid) || pid <= 0) throw new Error('Invalid run.lock; inspect it before removing it.');
    try { process.kill(pid, 0); }
    catch (probe) {
      if ((probe as NodeJS.ErrnoException).code === 'ESRCH') {
        fs.unlinkSync(file);
        return acquireLock(directory);
      }
      throw probe;
    }
    throw new Error(`Run is active in process ${pid}. Wait for it or stop that process before resuming.`);
  }
  return () => fs.unlinkSync(file);
}
